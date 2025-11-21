package driver

import (
	"sync"

	"github.com/sarchlab/akita/v4/mem/vm"
)

type buffer struct {
	vAddr Ptr
	size  uint64
	freed bool

	// After a kernel is launched, the l2 cache contain dirty data that belongs
	// to this buffer. Therefore, copying from or to this buffer triggers L2
	// flushing.
	l2Dirty bool
}

// Context is an opaque struct that carries the information used by the driver.
type Context struct {
	pid           vm.PID
	currentGPUID  int
	prevPageVAddr uint64
	l2Dirty       bool

	queueMutex sync.Mutex
	queues     []*CommandQueue

	buffersMutex sync.Mutex // Add a mutex lock to the Context structure to protect concurrent access to buffers.
	buffers []*buffer
}

func (c *Context) markAllBuffersDirty() {
	for _, b := range c.buffers {
		b.l2Dirty = true
	}
}

func (c *Context) markAllBuffersClean() {
	for _, b := range c.buffers {
		b.l2Dirty = false
	}
}

func (c *Context) removeFreedBuffers() {
	c.buffersMutex.Lock()
    defer c.buffersMutex.Unlock()

	newBuffers := make([]*buffer, 0, len(c.buffers))
    for _, b := range c.buffers {
        if !b.freed {
            newBuffers = append(newBuffers, b)
        }
    }
    c.buffers = newBuffers
}
