package interfaces

import (
	"image"
	"github.com/petar/GoMNIST"
)

type INet interface {
	Init()
	GetSize() []int
	GetInput() []float64
	GetLabel() []float64
	SetOutput([]float64)
	SetError(float64)
	Train(TrainParams, *GoMNIST.Set)
	Test(image.Image)
}
