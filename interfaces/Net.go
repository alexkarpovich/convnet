package interfaces

import (
	"github.com/alexkarpovich/convnet/config"
	"github.com/petar/GoMNIST"
)

type INet interface {
	Init()
	FromConfig(config.Net) chan NetState
	Size() []int
	Input() []float64
	Label() []float64
	State() NetState
	SetError(float64)
	SetOutput([]float64)
	Weights() []WeightsState
	LoadWeights([]WeightsState)
	Train(TrainParams, *GoMNIST.Set)
	StopTraining()
	Test([]byte) []float64
	IsTraining() bool
	LearningRate() float64
	MaxIterations() int
	MinError() float64
}

type NetState struct {
	Size []int `json:"size"`
	In []float64 `json:"in"`
	Out []float64 `json:"out"`
	Error float64 `json:"error"`
	Iteration int `json:"iteration"`
	Layers []LayerState `json:"layers"`
}

