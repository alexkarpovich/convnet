package layers

import (
	//"fmt"
	"math"
	"math/rand"
	. "github.com/alexkarpovich/convnet/utils"
	"github.com/alexkarpovich/convnet/interfaces"
)

type OutputLayer struct {
	*Layer
	weights [][]float64
	deltas []float64
}

func (l *OutputLayer) Prepare() {
	inSize := l.prev.Prop("outSize").([]int)
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
	prevOut := l.prev.Prop("out").([]float64)
	prevSize := l.prev.Prop("outSize").([]int)

	for j:=0; j<l.size[0]; j++ {
		s := 0.0

		for i:=0; i<prevSize[0]; i++ {
			s += prevOut[i] * l.weights[i][j]
		}

		l.in[j] = s
		l.out[j] = Sigmoid(s)
	}

	l.net.SetOutput(l.out)
	l.net.SetError(l.error())
}

func (l *OutputLayer) BackProp() {
	label := l.net.Label();
	inSize := l.prev.Prop("outSize").([]int)
	prevOut := l.prev.Prop("out").([]float64)

	for i:=0; i<l.size[0]; i++ {
		l.deltas[i] = label[i] - l.out[i];
	}

	for j:=0; j<l.size[0]; j++ {
		for i:=0; i<inSize[0];i++ {
			l.weights[i][j] += l.net.LearningRate()*l.deltas[j]*DSigmoid(l.in[j])*prevOut[i];
		}
	}
}

func (l *OutputLayer) Prop(name string) interface{} {
	switch name {
	case "outSize": return l.size
	case "in": return l.in
	case "deltas": return l.deltas
	case "weights": return l.weights
	}

	return nil
}

func (l *OutputLayer) error() float64 {
	err := 0.0

	for i := range l.deltas {
		err += math.Pow(l.deltas[i], 2)
	}

	return err
}

func (l *OutputLayer) State() interfaces.LayerState {
	out := make([]float64, len(l.out))
	copy(out, l.out)
	state := interfaces.LayerState{
		Class:l.class,
		Size: l.size,
		Out: out,
	}

	return state
}

func (l *OutputLayer) WeightsState() interfaces.WeightsState {
	w := make([][]float64, len(l.weights))
	for i:=0; i<len(l.weights);i++ {
		w[i] = make([]float64, len(l.weights[0]))
		copy(w[i], l.weights[i])
	}

	weightsState := interfaces.WeightsState{
		Class: l.class,
		Weights: w,
	}

	return weightsState
}

func (l *OutputLayer) SetWeightsState(weightsState interfaces.WeightsState) {
	l.weights = weightsState.Weights
}
