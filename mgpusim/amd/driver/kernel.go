package driver

import (
	"encoding/binary"
	"log"
	"reflect"
	"bytes"
	"fmt"
	
	"github.com/sarchlab/akita/v4/sim"
	"github.com/sarchlab/mgpusim/v4/amd/driver/internal"
	"github.com/sarchlab/mgpusim/v4/amd/insts"
	"github.com/sarchlab/mgpusim/v4/amd/kernels"
)

// EnqueueLaunchKernel schedules kernel to be launched later
func (d *Driver) EnqueueLaunchKernel(
	queue *CommandQueue,
	co *insts.HsaCo,
	gridSize [3]uint32,
	wgSize [3]uint16,
	kernelArgs interface{},
) (dCoData, dKernArgData, dPacket Ptr) {
	dev := d.devices[queue.GPUID]

	// ---- 处理 Unified GPU 情况 ----
	if dev.Type == internal.DeviceTypeUnifiedGPU {
		// 可以暂时打印警告而非 panic
		log.Printf("Unified GPU kernel execution is experimental, skipping panic.")
		d.enqueueLaunchUnifiedKernel(queue, co, gridSize, wgSize, kernelArgs)
		return 0, 0, 0
	}

	// ---- 普通 GPU 设备 ----
	dCoData, dKernArgData, dPacket = d.allocateGPUMemory(queue.Context, co)

	packet := d.createAQLPacket(gridSize, wgSize, dCoData, dKernArgData)
	buf := new(bytes.Buffer)
	err := binary.Write(buf, binary.LittleEndian, packet)
	if err != nil {
		panic(err)
	}
	packetBytes := buf.Bytes()
	newKernelArgs := d.prepareLocalMemory(co, kernelArgs, packetBytes)

	// 将 kernel 对象、参数、AQL 包传入 GPU
	d.EnqueueMemCopyH2D(queue, dCoData, co.Data)
	d.EnqueueMemCopyH2D(queue, dKernArgData, newKernelArgs)
	d.EnqueueMemCopyH2D(queue, dPacket, packet)

	// 启动 kernel
	d.enqueueLaunchKernelCommand(queue, co, packet, dPacket)


	return dCoData, dKernArgData, dPacket
}


func (d *Driver) allocateGPUMemory(
	ctx *Context,
	co *insts.HsaCo,
) (dCoData, dKernArgData, dPacket Ptr) {
	dCoData = d.AllocateMemory(ctx, uint64(len(co.Data)))
	dKernArgData = d.AllocateMemory(ctx, co.KernargSegmentByteSize)

	packet := kernels.HsaKernelDispatchPacket{}
	dPacket = d.AllocateMemory(ctx, uint64(binary.Size(packet)))

	log.Printf("[allocateGPUMemory] dCoData: 0x%x, dKernArgData: 0x%x, dPacket: 0x%x\n",
		dCoData, dKernArgData, dPacket)
	return dCoData, dKernArgData, dPacket
}

func (d *Driver) prepareLocalMemory(co *insts.HsaCo, kernelArgs interface{}, packet []byte) []byte {
    // helper: write a value to buffer using little endian, handling common kinds
    writeValue := func(buf *bytes.Buffer, v reflect.Value) error {
        // unwrap interface
        if v.Kind() == reflect.Interface {
            v = v.Elem()
        }
        switch v.Kind() {
        case reflect.Uintptr, reflect.Uint64:
            return binary.Write(buf, binary.LittleEndian, uint64(v.Uint()))
        case reflect.Uint32:
            return binary.Write(buf, binary.LittleEndian, uint32(v.Uint()))
        case reflect.Int, reflect.Int64:
            return binary.Write(buf, binary.LittleEndian, int64(v.Int()))
        case reflect.Int32:
            return binary.Write(buf, binary.LittleEndian, int32(v.Int()))
        case reflect.Int16:
            return binary.Write(buf, binary.LittleEndian, int16(v.Int()))
        case reflect.Int8:
            return binary.Write(buf, binary.LittleEndian, int8(v.Int()))
        case reflect.Float32:
            return binary.Write(buf, binary.LittleEndian, float32(v.Float()))
        case reflect.Float64:
            return binary.Write(buf, binary.LittleEndian, float64(v.Float()))
        default:
            // If it's a named type like driver.Ptr (alias of uint64), treat by kind underlying
            if v.CanConvert(reflect.TypeOf(uint64(0))) {
                return binary.Write(buf, binary.LittleEndian, uint64(v.Convert(reflect.TypeOf(uint64(0))).Uint()))
            }
            return fmt.Errorf("unsupported arg kind: %v", v.Kind())
        }
    }

    var buf bytes.Buffer

    if kernelArgs == nil {
        // No args: return empty slice of appropriate size (maybe still need packet)
        return buf.Bytes()
    }

    rv := reflect.ValueOf(kernelArgs)
    rt := rv.Type()

    // Case 1: direct []byte -> use as-is
    if b, ok := kernelArgs.([]byte); ok {
        log.Printf("prepareLocalMemory: kernelArgs is []byte len=%d", len(b))
        return b
    }

    // Case 2: []interface{} or slice of concrete types -> iterate elements
    if rv.Kind() == reflect.Slice {
        log.Printf("prepareLocalMemory: kernelArgs is slice kind=%v elem=%v len=%d", rt.Kind(), rt.Elem(), rv.Len())
        for i := 0; i < rv.Len(); i++ {
            elem := rv.Index(i)
            // handle pointer-to-type inside slice (e.g. driver.Ptr as uint64)
            if err := writeValue(&buf, elem); err != nil {
                log.Printf("prepareLocalMemory: writeValue failed for element %d: %v", i, err)
                // try to be permissive: if element is a slice/array of bytes, write raw
                if b, ok := elem.Interface().([]byte); ok {
                    buf.Write(b)
                    continue
                }
                panic(err)
            }
        }
    } else if rv.Kind() == reflect.Ptr && rv.Elem().Kind() == reflect.Struct {
        // Case 3: pointer to struct -> iterate fields in order and write
        log.Printf("prepareLocalMemory: kernelArgs is *struct %v", rt.Elem().Name())
        rv = rv.Elem()
        rt = rv.Type()
        for i := 0; i < rv.NumField(); i++ {
            fld := rv.Field(i)
            if !fld.CanInterface() {
                // unexported field — still we can try to read if addressable
            }
            if err := writeValue(&buf, fld); err != nil {
                log.Printf("prepareLocalMemory: writeValue failed for field %s: %v", rt.Field(i).Name, err)
                panic(err)
            }
        }
    } else {
        // Last resort: attempt to directly write the value
        log.Printf("prepareLocalMemory: kernelArgs kind=%v (fallback)", rv.Kind())
        if err := writeValue(&buf, rv); err != nil {
            log.Printf("prepareLocalMemory: unsupported kernelArgs type: %v", rt)
            panic(err)
        }
    }

    out := buf.Bytes()
    log.Printf("prepareLocalMemory: serialized kernel-args len=%d bytes", len(out))

    // safety cap: refuse absurd sizes
    const maxArgBytes = 1 << 30 // 1 GB
    if uint64(len(out)) > uint64(maxArgBytes) {
        log.Fatalf("prepareLocalMemory: kernel-args too large: %d bytes", len(out))
    }

    return out
}

// LaunchKernel is an easy way to run a kernel on the GCN3 simulator. It
// launches the kernel immediately.
func (d *Driver) LaunchKernel(
	ctx *Context,
	co *insts.HsaCo,
	gridSize [3]uint32,
	wgSize [3]uint16,
	kernelArgs interface{},
) {
	queue := d.CreateCommandQueue(ctx)
	// d.EnqueueLaunchKernel(queue, co, gridSize, wgSize, kernelArgs)
	dCoData, dKernArgData, dPacket := d.EnqueueLaunchKernel(queue, co, gridSize, wgSize, kernelArgs)
	d.DrainCommandQueue(queue)
	log.Printf("[LaunchKernel-Free] dCoData: 0x%x, dKernArgData: 0x%x, dPacket: 0x%x\n",
		dCoData, dKernArgData, dPacket)
	d.FreeMemory(ctx, dCoData)
	d.FreeMemory(ctx, dKernArgData)
	d.FreeMemory(ctx, dPacket)
}

func (d *Driver) createAQLPacket(
	gridSize [3]uint32,
	wgSize [3]uint16,
	dCoData Ptr,
	dKernArgData Ptr,
) *kernels.HsaKernelDispatchPacket {
	packet := new(kernels.HsaKernelDispatchPacket)
	packet.GridSizeX = gridSize[0]
	packet.GridSizeY = gridSize[1]
	packet.GridSizeZ = gridSize[2]
	packet.WorkgroupSizeX = wgSize[0]
	packet.WorkgroupSizeY = wgSize[1]
	packet.WorkgroupSizeZ = wgSize[2]
	packet.KernelObject = uint64(dCoData)
	packet.KernargAddress = uint64(dKernArgData)
	
	return packet
}

func (d *Driver) enqueueLaunchKernelCommand(
	queue *CommandQueue,
	co *insts.HsaCo,
	packet *kernels.HsaKernelDispatchPacket,
	dPacket Ptr,
) {
	cmd := &LaunchKernelCommand{
		ID:         sim.GetIDGenerator().Generate(),
		CodeObject: co,
		DPacket:    dPacket,
		Packet:     packet,
	}
	d.Enqueue(queue, cmd)
}

func (d *Driver) enqueueLaunchUnifiedKernelCommand(
	queue *CommandQueue,
	co *insts.HsaCo,
	packet []*kernels.HsaKernelDispatchPacket,
	dPacket []Ptr,
) {
	cmd := &LaunchUnifiedMultiGPUKernelCommand{
		ID:           sim.GetIDGenerator().Generate(),
		CodeObject:   co,
		DPacketArray: dPacket,
		PacketArray:  packet,
	}
	d.Enqueue(queue, cmd)
}
func (d *Driver) enqueueLaunchUnifiedKernel(
	queue *CommandQueue,
	co *insts.HsaCo,
	gridSize [3]uint32,
	wgSize [3]uint16,
	kernelArgs interface{},
) {
	dev := d.devices[queue.GPUID]
	initGPUID := queue.Context.currentGPUID
	queueArray := make([]*CommandQueue, len(dev.UnifiedGPUIDs)+1)
	dCoDataArray := make([]Ptr, len(dev.UnifiedGPUIDs)+1)
	dKernArgDataArray := make([]Ptr, len(dev.UnifiedGPUIDs)+1)
	dPacketArray := make([]Ptr, len(dev.UnifiedGPUIDs)+1)
	packetArray := make([]*kernels.HsaKernelDispatchPacket, len(dev.UnifiedGPUIDs)+1)
	// fmt.Printf("# of GPUs : %v \n", len(dev.UnifiedGPUIDs))

	for i, gpuID := range dev.UnifiedGPUIDs {
		queueArray[i] = queue
		queueArray[i].Context.currentGPUID = gpuID
		dCoData, dKernArgData, dPacket := d.allocateGPUMemory(queue.Context, co)

		packet := d.createAQLPacket(gridSize, wgSize, dCoData, dKernArgData)
		buf := new(bytes.Buffer)
		err := binary.Write(buf, binary.LittleEndian, packet)
		if err != nil {
			panic(err)
		}
		packetBytes := buf.Bytes()
		newKernelArgs := d.prepareLocalMemory(co, kernelArgs, packetBytes)

		d.EnqueueMemCopyH2D(queue, dCoData, co.Data)
		d.EnqueueMemCopyH2D(queue, dKernArgData, newKernelArgs)
		d.EnqueueMemCopyH2D(queue, dPacket, packet)

		dCoDataArray[i] = dCoData
		dKernArgDataArray[i] = dKernArgData
		dPacketArray[i] = dPacket
		packetArray[i] = packet
		// fmt.Printf("packetArray: %v \n", packetArray[i])
	}

	queue.Context.currentGPUID = initGPUID
	d.enqueueLaunchUnifiedKernelCommand(queue, co, packetArray, dPacketArray)
}

func (d *Driver) lazyPrepareLocalMemory(
	co *insts.HsaCo,
	kernelArgs interface{},
) (newKernelArgs interface{}, ldsSize uint32) {
	newKernelArgs = reflect.New(reflect.TypeOf(kernelArgs).Elem()).Interface()
	reflect.ValueOf(newKernelArgs).Elem().
		Set(reflect.ValueOf(kernelArgs).Elem())

	ldsSize = co.WGGroupSegmentByteSize

	if reflect.TypeOf(newKernelArgs).Kind() == reflect.Slice {
		// From server, do nothing
	} else {
		kernArgStruct := reflect.ValueOf(newKernelArgs).Elem()
		for i := 0; i < kernArgStruct.NumField(); i++ {
			arg := kernArgStruct.Field(i).Interface()

			switch ldsPtr := arg.(type) {
			case LocalPtr:
				kernArgStruct.Field(i).SetUint(uint64(ldsSize))
				ldsSize += uint32(ldsPtr)
			}
		}
	}

	return newKernelArgs, ldsSize
}

// LazyEnqueueLaunchKernel schedules kernel to be launched later
func (d *Driver) LazyEnqueueLaunchKernel(
	queue *CommandQueue,
	co *insts.HsaCo,
	gridSize [3]uint32,
	wgSize [3]uint16,
	kernelArgs interface{},
) (dCoData, dKernArgData, dPacket Ptr) {
	dev := d.devices[queue.GPUID]

	if dev.Type == internal.DeviceTypeUnifiedGPU {
		panic("unified kernel not supported yet")
	} else {
		// 1. Allocate and copy dCoData
		d.LazyEnqueueMemCopyH2D(queue, co.Data, uint64(len(co.Data)))
		d.DrainCommandQueue(queue)
		dCoData = d.AllocatedVAddr
		
		// 2. Allocate and copy dKernArgData
		newKernelArgs, ldsSize := d.lazyPrepareLocalMemory(co, kernelArgs)
		d.LazyEnqueueMemCopyH2D(queue, newKernelArgs, co.KernargSegmentByteSize)
		d.DrainCommandQueue(queue)
		dKernArgData = d.AllocatedVAddr

		// 3. Allocate and copy dPacket
		packet := d.createAQLPacket(gridSize, wgSize, dCoData, dKernArgData)
		packet.GroupSegmentSize = ldsSize
		d.LazyEnqueueMemCopyH2D(queue, packet, uint64(binary.Size(packet)))
		d.DrainCommandQueue(queue)
		dPacket = d.AllocatedVAddr
		
		d.enqueueLaunchKernelCommand(queue, co, packet, dPacket)

		log.Printf("[LazyEnqueueLaunchKernel-Allocate] dCoData: 0x%x, dKernArgData: 0x%x, dPacket: 0x%x\n",
			dCoData, dKernArgData, dPacket)

		return dCoData, dKernArgData, dPacket
	}
}

// LazyLaunchKernel is an easy way to run a kernel on the GCN3 simulator. 
// It launches the kernel immediately.
// Compared to LaunchKernel, LazyLaunchKernel lazily allocates memory.
func (d *Driver) LazyLaunchKernel(
	ctx *Context,
	co *insts.HsaCo,
	gridSize [3]uint32,
	wgSize [3]uint16,
	kernelArgs interface{},
) {
	queue := d.CreateCommandQueue(ctx)
	dCoData, dKernArgData, dPacket := d.LazyEnqueueLaunchKernel(queue, co, gridSize, wgSize, kernelArgs)
	d.DrainCommandQueue(queue)
	log.Printf("[LazyLaunchKernel-Free] dCoData: 0x%x, dKernArgData: 0x%x, dPacket: 0x%x\n",
		dCoData, dKernArgData, dPacket)
	d.FreeMemory(ctx, dCoData)
	d.FreeMemory(ctx, dKernArgData)
	d.FreeMemory(ctx, dPacket)
}