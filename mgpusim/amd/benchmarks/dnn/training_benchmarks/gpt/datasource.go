package gpt

import (
	"fmt"

	"github.com/sarchlab/mgpusim/v4/amd/benchmarks/dnn/tensor"
)

// DataSource defines the training dataset for the xor operation.
type DataSource struct {
	to        tensor.Operator
	allData   []float64
	allLabel  []int
	contextLength int
	currPtr   int
}

// NewDataSource creates a new XOR datasource
func NewDataSource(to tensor.Operator, seq []int, contextLength int) *DataSource {
	ds := &DataSource{
		contextLength: contextLength,
		to:        to,
	}
	ds.tokenSeqToTensor(seq)
	return ds
}

// NextBatch returns the next batch data.
func (ds *DataSource) NextBatch(batchSize int) (
	data tensor.Tensor,
	label []int,
) {
	start := ds.currPtr
	end := start + batchSize

	if end > len(ds.allLabel) {
		end = len(ds.allLabel)
	}

	if start == end {
		return nil, nil
	}

	rawData := ds.allData[start*ds.contextLength : end*ds.contextLength]
	data = ds.to.CreateWithData(rawData, []int{end - start, ds.contextLength}, "")

	label = ds.allLabel[start:end]

	ds.currPtr = end

	return data, label
}

// Rewind moves the pointer to the beginning of the training set.
func (ds *DataSource) Rewind() {
	ds.currPtr = 0
}

// NextBatch returns the next batch of data.
func (ds *DataSource) LazyNextBatch(batchSize int) (
	data tensor.Tensor,
	label []int,
) {
	start := ds.currPtr
	end := start + batchSize

	if end > len(ds.allLabel) {
		end = len(ds.allLabel)
	}

	rawData := ds.allData[start*ds.contextLength : end*ds.contextLength]
	data = ds.to.LazyCreateWithData(rawData, []int{end - start, ds.contextLength}, "")

	label = ds.allLabel[start:end]

	ds.currPtr = end

	return data, label
}

// tokenSeqToTensor converts a sequence into (X,Y) pairs
// X = context windows, Y = next token
func (ds *DataSource) tokenSeqToTensor(seq []int) {
	for i := 0; i+ds.contextLength < len(seq); i++ {
		// Take a window of length contextLength
		window := seq[i : i+ds.contextLength]
		for _, v := range window {
			ds.allData = append(ds.allData, float64(v))
		}
		// The next token is the label
		ds.allLabel = append(ds.allLabel, seq[i+ds.contextLength])

		fmt.Printf("example %2d: %v --> %d\n", i+1, window, seq[i+ds.contextLength])
	}
}