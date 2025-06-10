package mem

import (
	"github.com/sarchlab/akita/v4/mem/vm"
	"github.com/sarchlab/akita/v4/sim"
)

var accessReqByteOverhead = 12
var accessRspByteOverhead = 4
var controlMsgByteOverhead = 4

// AccessReq abstracts read and write requests that are sent to the
// cache modules or memory controllers.
type AccessReq interface {
	sim.Msg
	GetAddress() uint64
	GetByteSize() uint64
	GetPID() vm.PID
}

// A AccessRsp is a respond in the memory system.
type AccessRsp interface {
	sim.Msg
	sim.Rsp
}

// A ReadReq is a request sent to a memory controller to fetch data
type ReadReq struct {
	sim.MsgMeta

	Address            uint64
	AccessByteSize     uint64
	PID                vm.PID
	CanWaitForCoalesce bool
	Info               interface{}
}

// Meta returns the message meta.
func (r *ReadReq) Meta() *sim.MsgMeta {
	return &r.MsgMeta
}

// Clone returns cloned ReadReq with different ID
func (r *ReadReq) Clone() sim.Msg {
	cloneMsg := *r
	cloneMsg.ID = sim.GetIDGenerator().Generate()

	return &cloneMsg
}

// GenerateRsp generate DataReadyRsp to ReadReq
func (r *ReadReq) GenerateRsp(data []byte) sim.Rsp {
	rsp := DataReadyRspBuilder{}.
		WithSrc(r.Dst).
		WithDst(r.Src).
		WithRspTo(r.ID).
		WithData(data).
		Build()

	return rsp
}

// GetByteSize returns the number of byte that the request is accessing.
func (r *ReadReq) GetByteSize() uint64 {
	return r.AccessByteSize
}

// GetAddress returns the address that the request is accessing
func (r *ReadReq) GetAddress() uint64 {
	return r.Address
}

// GetPID returns the process ID that the request is working on.
func (r *ReadReq) GetPID() vm.PID {
	return r.PID
}

// ReadReqBuilder can build read requests.
type ReadReqBuilder struct {
	src, dst           sim.RemotePort
	pid                vm.PID
	address, byteSize  uint64
	canWaitForCoalesce bool
	info               interface{}
}

// WithSrc sets the source of the request to build.
func (b ReadReqBuilder) WithSrc(src sim.RemotePort) ReadReqBuilder {
	b.src = src
	return b
}

// WithDst sets the destination of the request to build.
func (b ReadReqBuilder) WithDst(dst sim.RemotePort) ReadReqBuilder {
	b.dst = dst
	return b
}

// WithPID sets the PID of the request to build.
func (b ReadReqBuilder) WithPID(pid vm.PID) ReadReqBuilder {
	b.pid = pid
	return b
}

// WithInfo sets the Info of the request to build.
func (b ReadReqBuilder) WithInfo(info interface{}) ReadReqBuilder {
	b.info = info
	return b
}

// WithAddress sets the address of the request to build.
func (b ReadReqBuilder) WithAddress(address uint64) ReadReqBuilder {
	b.address = address
	return b
}

// WithByteSize sets the byte size of the request to build.
func (b ReadReqBuilder) WithByteSize(byteSize uint64) ReadReqBuilder {
	b.byteSize = byteSize
	return b
}

// CanWaitForCoalesce allow the request to build to wait for coalesce.
func (b ReadReqBuilder) CanWaitForCoalesce() ReadReqBuilder {
	b.canWaitForCoalesce = true
	return b
}

// Build creates a new ReadReq
func (b ReadReqBuilder) Build() *ReadReq {
	r := &ReadReq{}
	r.ID = sim.GetIDGenerator().Generate()
	r.Src = b.src
	r.Dst = b.dst
	r.TrafficBytes = accessReqByteOverhead
	r.Address = b.address
	r.PID = b.pid
	r.Info = b.info
	r.AccessByteSize = b.byteSize
	r.CanWaitForCoalesce = b.canWaitForCoalesce

	return r
}

// A WriteReq is a request sent to a memory controller to write data
type WriteReq struct {
	sim.MsgMeta

	Address            uint64
	Data               []byte
	DirtyMask          []bool
	PID                vm.PID
	CanWaitForCoalesce bool
	Info               interface{}
}

// Meta returns the meta data attached to a request.
func (r *WriteReq) Meta() *sim.MsgMeta {
	return &r.MsgMeta
}

// Clone returns cloned WriteReq with different ID
func (r *WriteReq) Clone() sim.Msg {
	cloneMsg := *r
	cloneMsg.ID = sim.GetIDGenerator().Generate()

	return &cloneMsg
}

// GenerateRsp generate WriteDoneRsp to the original WriteReq
func (r *WriteReq) GenerateRsp() sim.Rsp {
	rsp := WriteDoneRspBuilder{}.
		WithSrc(r.Dst).
		WithDst(r.Src).
		WithRspTo(r.ID).
		Build()

	return rsp
}

// GetByteSize returns the number of byte that the request is writing.
func (r *WriteReq) GetByteSize() uint64 {
	return uint64(len(r.Data))
}

// GetAddress returns the address that the request is accessing
func (r *WriteReq) GetAddress() uint64 {
	return r.Address
}

// GetPID returns the PID of the read address
func (r *WriteReq) GetPID() vm.PID {
	return r.PID
}

// WriteReqBuilder can build read requests.
type WriteReqBuilder struct {
	src, dst           sim.RemotePort
	pid                vm.PID
	info               interface{}
	address            uint64
	data               []byte
	dirtyMask          []bool
	canWaitForCoalesce bool
}

// WithSrc sets the source of the request to build.
func (b WriteReqBuilder) WithSrc(src sim.RemotePort) WriteReqBuilder {
	b.src = src
	return b
}

// WithDst sets the destination of the request to build.
func (b WriteReqBuilder) WithDst(dst sim.RemotePort) WriteReqBuilder {
	b.dst = dst
	return b
}

// WithPID sets the PID of the request to build.
func (b WriteReqBuilder) WithPID(pid vm.PID) WriteReqBuilder {
	b.pid = pid
	return b
}

// WithInfo sets the information attached to the request to build.
func (b WriteReqBuilder) WithInfo(info interface{}) WriteReqBuilder {
	b.info = info
	return b
}

// WithAddress sets the address of the request to build.
func (b WriteReqBuilder) WithAddress(address uint64) WriteReqBuilder {
	b.address = address
	return b
}

// WithData sets the data of the request to build.
func (b WriteReqBuilder) WithData(data []byte) WriteReqBuilder {
	b.data = data
	return b
}

// WithDirtyMask sets the dirty mask of the request to build.
func (b WriteReqBuilder) WithDirtyMask(mask []bool) WriteReqBuilder {
	b.dirtyMask = mask
	return b
}

// CanWaitForCoalesce allow the request to build to wait for coalesce.
func (b WriteReqBuilder) CanWaitForCoalesce() WriteReqBuilder {
	b.canWaitForCoalesce = true
	return b
}

// Build creates a new WriteReq
func (b WriteReqBuilder) Build() *WriteReq {
	r := &WriteReq{}
	r.ID = sim.GetIDGenerator().Generate()
	r.Src = b.src
	r.Dst = b.dst
	r.PID = b.pid
	r.Info = b.info
	r.Address = b.address
	r.Data = b.data
	r.TrafficBytes = len(r.Data) + accessReqByteOverhead
	r.DirtyMask = b.dirtyMask
	r.CanWaitForCoalesce = b.canWaitForCoalesce

	return r
}

// A DataReadyRsp is the respond sent from the lower module to the higher
// module that carries the data loaded.
type DataReadyRsp struct {
	sim.MsgMeta

	RespondTo string // The ID of the request it replies
	Data      []byte
}

// Meta returns the meta data attached to each message.
func (r *DataReadyRsp) Meta() *sim.MsgMeta {
	return &r.MsgMeta
}

// Clone returns cloned DataReadyRsp with different ID
func (r *DataReadyRsp) Clone() sim.Msg {
	cloneMsg := *r
	cloneMsg.ID = sim.GetIDGenerator().Generate()

	return &cloneMsg
}

// GetRspTo returns the ID if the request that the respond is responding to.
func (r *DataReadyRsp) GetRspTo() string {
	return r.RespondTo
}

// DataReadyRspBuilder can build data ready responds.
type DataReadyRspBuilder struct {
	src, dst sim.RemotePort
	rspTo    string
	data     []byte
}

// WithSrc sets the source of the request to build.
func (b DataReadyRspBuilder) WithSrc(src sim.RemotePort) DataReadyRspBuilder {
	b.src = src
	return b
}

// WithDst sets the destination of the request to build.
func (b DataReadyRspBuilder) WithDst(dst sim.RemotePort) DataReadyRspBuilder {
	b.dst = dst
	return b
}

// WithRspTo sets ID of the request that the respond to build is replying to.
func (b DataReadyRspBuilder) WithRspTo(id string) DataReadyRspBuilder {
	b.rspTo = id
	return b
}

// WithData sets the data of the request to build.
func (b DataReadyRspBuilder) WithData(data []byte) DataReadyRspBuilder {
	b.data = data
	return b
}

// Build creates a new DataReadyRsp
func (b DataReadyRspBuilder) Build() *DataReadyRsp {
	r := &DataReadyRsp{}
	r.ID = sim.GetIDGenerator().Generate()
	r.Src = b.src
	r.Dst = b.dst
	r.TrafficBytes = len(b.data) + accessRspByteOverhead
	r.RespondTo = b.rspTo
	r.Data = b.data

	return r
}

// A WriteDoneRsp is a respond sent from the lower module to the higher module
// to mark a previous requests is completed successfully.
type WriteDoneRsp struct {
	sim.MsgMeta

	RespondTo string
}

// Meta returns the meta data associated with the message.
func (r *WriteDoneRsp) Meta() *sim.MsgMeta {
	return &r.MsgMeta
}

// Clone returns cloned WriteDoneRsp with different ID
func (r *WriteDoneRsp) Clone() sim.Msg {
	cloneMsg := *r
	cloneMsg.ID = sim.GetIDGenerator().Generate()

	return &cloneMsg
}

// GetRspTo returns the ID of the request that the respond is responding to.
func (r *WriteDoneRsp) GetRspTo() string {
	return r.RespondTo
}

// WriteDoneRspBuilder can build data ready responds.
type WriteDoneRspBuilder struct {
	src, dst sim.RemotePort
	rspTo    string
}

// WithSrc sets the source of the request to build.
func (b WriteDoneRspBuilder) WithSrc(src sim.RemotePort) WriteDoneRspBuilder {
	b.src = src
	return b
}

// WithDst sets the destination of the request to build.
func (b WriteDoneRspBuilder) WithDst(dst sim.RemotePort) WriteDoneRspBuilder {
	b.dst = dst
	return b
}

// WithRspTo sets ID of the request that the respond to build is replying to.
func (b WriteDoneRspBuilder) WithRspTo(id string) WriteDoneRspBuilder {
	b.rspTo = id
	return b
}

// Build creates a new WriteDoneRsp
func (b WriteDoneRspBuilder) Build() *WriteDoneRsp {
	r := &WriteDoneRsp{}
	r.ID = sim.GetIDGenerator().Generate()
	r.Src = b.src
	r.Dst = b.dst
	r.TrafficBytes = accessRspByteOverhead
	r.RespondTo = b.rspTo

	return r
}

// ControlMsg is the commonly used message type for controlling the components
// on the memory hierarchy. It is also used for resonpding the original
// requester with the Done field.
type ControlMsg struct {
	sim.MsgMeta

	DiscardTransations bool
	Restart            bool
	NotifyDone         bool
	Enable             bool
	Drain              bool
	Flush              bool
	Pause              bool
	Invalid            bool
}

// Meta returns the meta data assocated with the ControlMsg.
func (m *ControlMsg) Meta() *sim.MsgMeta {
	return &m.MsgMeta
}

// Clone returns cloned ControlMsg with different ID
func (m *ControlMsg) Clone() sim.Msg {
	cloneMsg := *m
	cloneMsg.ID = sim.GetIDGenerator().Generate()

	return &cloneMsg
}

// A ControlMsgBuilder can build control messages.
type ControlMsgBuilder struct {
	src, dst            sim.RemotePort
	discardTransactions bool
	restart             bool
	notifyDone          bool
	Enable              bool
	Drain               bool
	Flush               bool
	Pause               bool
	Invalid             bool
}

// WithSrc sets the source of the request to build.
func (b ControlMsgBuilder) WithSrc(src sim.RemotePort) ControlMsgBuilder {
	b.src = src
	return b
}

// WithDst sets the destination of the request to build.
func (b ControlMsgBuilder) WithDst(dst sim.RemotePort) ControlMsgBuilder {
	b.dst = dst
	return b
}

// ToDiscardTransactions sets the discard transactions bit of the control
// messages to 1.
func (b ControlMsgBuilder) ToDiscardTransactions() ControlMsgBuilder {
	b.discardTransactions = true
	return b
}

// ToRestart sets the restart bit of the control messages to 1.
func (b ControlMsgBuilder) ToRestart() ControlMsgBuilder {
	b.restart = true
	return b
}

// ToNotifyDone sets the "notify done" bit of the control messages to 1.
func (b ControlMsgBuilder) ToNotifyDone() ControlMsgBuilder {
	b.notifyDone = true
	return b
}

// WithCtrlInfo sets the enable bit of the control messages to 1.
func (b ControlMsgBuilder) WithCtrlInfo(
	enable bool, drain bool, flush bool, pause bool, invalid bool,
) ControlMsgBuilder {
	b.Enable = enable
	b.Drain = drain
	b.Flush = flush
	b.Pause = pause
	b.Invalid = invalid
	return b
}

// Build creates a new ControlMsg.
func (b ControlMsgBuilder) Build() *ControlMsg {
	m := &ControlMsg{}
	m.ID = sim.GetIDGenerator().Generate()
	m.Src = b.src
	m.Dst = b.dst
	m.TrafficBytes = controlMsgByteOverhead

	m.DiscardTransations = b.discardTransactions
	m.Restart = b.restart
	m.NotifyDone = b.notifyDone
	m.Enable = b.Enable
	m.Drain = b.Drain
	m.Flush = b.Flush
	m.Pause = b.Pause
	m.Invalid = b.Invalid

	return m
}

// A AllocateReq is a request sent to a memory allocator to allocate memory
type AllocateReq struct {
	sim.MsgMeta

	DeviceID uint64
	PID vm.PID
	ByteSize uint64
}

// Meta returns the message meta.
func (r *AllocateReq) Meta() *sim.MsgMeta {
	return &r.MsgMeta
}

// Clone returns cloned ReadReq with different ID
func (r *AllocateReq) Clone() sim.Msg {
	cloneMsg := *r
	cloneMsg.ID = sim.GetIDGenerator().Generate()

	return &cloneMsg
}

// GenerateRsp generate DataReadyRsp to ReadReq
func (r *AllocateReq) GenerateRsp(vAddr uint64) sim.Rsp {
	rsp := AllocateDoneRspBuilder{}.
		WithDeviceID(r.DeviceID).
		WithPID(r.PID).
		WithByteSize(r.ByteSize).
		WithVAddr(vAddr).
		WithRspTo(r.ID).
		Build()

	return rsp
}

type AllocateReqBuilder struct {
	deviceID uint64
	pid vm.PID
	byteSize uint64
}

// WithDeviceID sets the DeviceID of the request to build.
func (b AllocateReqBuilder) WithDeviceID(deviceID uint64) AllocateReqBuilder {
	b.deviceID = deviceID
	return b
}

// WithPID sets the PID of the request to build.
func (b AllocateReqBuilder) WithPID(pid vm.PID) AllocateReqBuilder {
	b.pid = pid
	return b
}

// WithByteSize sets the byte size of the request to build.
func (b AllocateReqBuilder) WithByteSize(byteSize uint64) AllocateReqBuilder {
	b.byteSize = byteSize
	return b
}

// Build creates a new AllocateReq
func (b AllocateReqBuilder) Build() *AllocateReq {
	r := &AllocateReq{}
	r.ID = sim.GetIDGenerator().Generate()
	r.DeviceID = b.deviceID
	r.PID = b.pid
	r.ByteSize = b.byteSize

	return r
}

type AllocateDoneRsp struct {
	sim.MsgMeta

	DeviceID uint64
	PID vm.PID
	ByteSize uint64

	VAddr uint64 // The start virtual address of the allocated memory
	RspTo string // The ID of the request that the respond is responding to
}

// Meta returns the message meta.
func (r *AllocateDoneRsp) Meta() *sim.MsgMeta {
	return &r.MsgMeta
}

// Clone returns cloned ReadReq with different ID
func (r *AllocateDoneRsp) Clone() sim.Msg {
	cloneMsg := *r
	cloneMsg.ID = sim.GetIDGenerator().Generate()

	return &cloneMsg
}

// GetRspTo returns the ID of the request that the respond is responding to.
func (r *AllocateDoneRsp) GetRspTo() string {
	return r.RspTo
}

type AllocateDoneRspBuilder struct {
	deviceID uint64
	pid vm.PID
	byteSize uint64

	vAddr uint64
	rspTo string
}

// WithDeviceID sets the DeviceID of the request to build.
func (b AllocateDoneRspBuilder) WithDeviceID(deviceID uint64) AllocateDoneRspBuilder {
	b.deviceID = deviceID
	return b
}

// WithPID sets the PID of the request to build.
func (b AllocateDoneRspBuilder) WithPID(pid vm.PID) AllocateDoneRspBuilder {
	b.pid = pid
	return b
}

// WithByteSize sets the byte size of the request to build.
func (b AllocateDoneRspBuilder) WithByteSize(byteSize uint64) AllocateDoneRspBuilder {
	b.byteSize = byteSize
	return b
}

// WithByteSize sets the virtual address of the request to build.
func (b AllocateDoneRspBuilder) WithVAddr(vAddr uint64) AllocateDoneRspBuilder {
	b.vAddr = vAddr
	return b
}

// WithRspTo sets ID of the request that the respond to build is replying to.
func (b AllocateDoneRspBuilder) WithRspTo(rspTo string) AllocateDoneRspBuilder {
	b.rspTo = rspTo
	return b
}

// Build creates a new AllocateReq
func (b AllocateDoneRspBuilder) Build() *AllocateDoneRsp {
	r := &AllocateDoneRsp{}
	r.ID = b.rspTo
	r.DeviceID = b.deviceID
	r.PID = b.pid
	r.ByteSize = b.byteSize
	r.VAddr = b.vAddr
	r.RspTo = b.rspTo

	return r
}

// A FreeReq is a request sent to a memory allocator to free memory
type FreeReq struct {
	sim.MsgMeta

	DeviceID uint64
	PID vm.PID
	VAddr uint64
}

// Meta returns the message meta.
func (r *FreeReq) Meta() *sim.MsgMeta {
	return &r.MsgMeta
}

// Clone returns cloned ReadReq with different ID
func (r *FreeReq) Clone() sim.Msg {
	cloneMsg := *r
	cloneMsg.ID = sim.GetIDGenerator().Generate()

	return &cloneMsg
}

// GenerateRsp generate DataReadyRsp to ReadReq
func (r *FreeReq) GenerateRsp(byteSize uint64) sim.Rsp {
	rsp := FreeDoneRspBuilder{}.
		WithDeviceID(r.DeviceID).
		WithPID(r.PID).
		WithVAddr(r.VAddr).
		WithByteSize(byteSize).
		WithRspTo(r.ID).
		Build()

	return rsp
}

type FreeReqBuilder struct {
	deviceID uint64
	pid vm.PID
	vAddr uint64
}

// WithDeviceID sets the DeviceID of the request to build.
func (b FreeReqBuilder) WithDeviceID(deviceID uint64) FreeReqBuilder {
	b.deviceID = deviceID
	return b
}

// WithPID sets the PID of the request to build.
func (b FreeReqBuilder) WithPID(pid vm.PID) FreeReqBuilder {
	b.pid = pid
	return b
}

// WithVAddr sets the virtual address of the request to build.
func (b FreeReqBuilder) WithVAddr(vAddr uint64) FreeReqBuilder {
	b.vAddr = vAddr
	return b
}

// Build creates a new FreeReq
func (b FreeReqBuilder) Build() *FreeReq {
	r := &FreeReq{}
	r.ID = sim.GetIDGenerator().Generate()
	r.DeviceID = b.deviceID
	r.PID = b.pid
	r.VAddr = b.vAddr

	return r
}

type FreeDoneRsp struct {
	sim.MsgMeta

	DeviceID uint64
	PID vm.PID
	VAddr uint64 // The start virtual address of the allocated memory
	
	ByteSize uint64
	RspTo string
}

// Meta returns the message meta.
func (r *FreeDoneRsp) Meta() *sim.MsgMeta {
	return &r.MsgMeta
}

// Clone returns cloned ReadReq with different ID
func (r *FreeDoneRsp) Clone() sim.Msg {
	cloneMsg := *r
	cloneMsg.ID = sim.GetIDGenerator().Generate()

	return &cloneMsg
}

// GetRspTo returns the ID of the request that the respond is responding to.
func (r *FreeDoneRsp) GetRspTo() string {
	return r.RspTo
}

type FreeDoneRspBuilder struct {
	deviceID uint64
	pid vm.PID
	byteSize uint64

	vAddr uint64
	rspTo string
}

// WithDeviceID sets the DeviceID of the request to build.
func (b FreeDoneRspBuilder) WithDeviceID(deviceID uint64) FreeDoneRspBuilder {
	b.deviceID = deviceID
	return b
}

// WithPID sets the PID of the request to build.
func (b FreeDoneRspBuilder) WithPID(pid vm.PID) FreeDoneRspBuilder {
	b.pid = pid
	return b
}

// WithByteSize sets the virtual address of the request to build.
func (b FreeDoneRspBuilder) WithVAddr(vAddr uint64) FreeDoneRspBuilder {
	b.vAddr = vAddr
	return b
}

// WithByteSize sets the byte size of the request to build.
func (b FreeDoneRspBuilder) WithByteSize(byteSize uint64) FreeDoneRspBuilder {
	b.byteSize = byteSize
	return b
}

// WithRspTo sets ID of the request that the respond to build is replying to.
func (b FreeDoneRspBuilder) WithRspTo(rspTo string) FreeDoneRspBuilder {
	b.rspTo = rspTo
	return b
}

// Build creates a new FreeReq
func (b FreeDoneRspBuilder) Build() *FreeDoneRsp {
	r := &FreeDoneRsp{}
	r.ID = b.rspTo
	r.DeviceID = b.deviceID
	r.PID = b.pid
	r.VAddr = b.vAddr
	r.ByteSize = b.byteSize
	r.RspTo = b.rspTo

	return r
}
