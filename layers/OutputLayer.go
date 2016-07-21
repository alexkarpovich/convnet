package layers

import (
	//"fmt"
	"math"
	"math/rand"
	. "github.com/alexkarpovich/convnet/api/convnet/utils"
)

type OutputLayer struct {
	*Layer
	weights [][]float64
	deltas []float64
}

func (l *OutputLayer) Prepare() {
	inSize := l.prev.GetProp("outSize").([]int)
	length := inSize[0]

	l.weights = make([][]float64, length)
	for i:=0; i<length; i++ {
		l.weights[i] = make([]float64, l.size[0])

		for j:=0; j<l.size[0]; j++ {
			l.weights[i][j] = rand.Float64()*2-1
		}
	}

	l.in = make([]float64, l.size[0])
	l.out = make([]float64, l.size[0])
	l.deltas = make([]float64, l.size[0])
}

func (l *OutputLayer) FeedForward() {
	prevOut := l.prev.GetProp("out").([]float64)
	prevSize := l.prev.GetProp("outSize").([]int)

	for j:=0; j<l.size[0]; j++ {
		s := 0.0

		for i:=0; i<prevSize[0]; i++ {
			s += prevOut[i] * l.weights[i][j]
		}

		l.in[j] = s
		l.out[j] = Sigmoid(s)
	}

	l.net.SetOutput(l.out)
	l.net.SetError(l.GetError())
}

func (l *OutputLayer) BackProp() {
	alpha := 0.001
	label := l.net.GetLabel();
	inSize := l.prev.GetProp("outSize").([]int)
	prevOut := l.prev.GetProp("out").([]float64)

	for i:=0; i<l.size[0]; i++ {
		l.deltas[i] = label[i] - l.out[i];
	}

	for j:=0; j<l.size[0]; j++ {
		for i:=0; i<inSize[0];i++ {
			l.weights[i][j] += alpha*l.deltas[j]*DSigmoid(l.in[j])*prevOut[i];
		}
	}
}

func (l *OutputLayer) GetProp(name string) interface{} {
	switch name {
	case "outSize": return l.size
	case "in": return l.in
	case "deltas": return l.deltas
	case "weights": return l.weights
	}

	return nil
}

func (l *OutputLayer) GetError() float64 {
	err := 0.0

	for i := range l.deltas {
		err += math.Pow(l.deltas[i], 2)
	}

	return err
}
