package layers

import (
	//"fmt"
	"math/rand"
	. "github.com/alexkarpovich/convnet/utils"
	"github.com/alexkarpovich/convnet/interfaces"
)

type FCLayer struct {
	*Layer
	weights [][]float64
	deltas []float64
}

func (l *FCLayer) Prepare() {
	inSize := l.prev.Prop("outSize").([]int)
	count := l.prev.Prop("count").(int)
	length := count * inSize[0] * inSize[1]

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

func (l *FCLayer) FeedForward() {
	prevOut := l.prev.Prop("out").([]float64)
	prevLength := len(prevOut)

	for j:=0; j<l.size[0]; j++ {
		s := 0.0

		for i:=0; i<prevLength; i++ {
			s += prevOut[i] * l.weights[i][j]
		}

		l.in[j] = s
		l.out[j] = Sigmoid(s)
	}
}

func (l *FCLayer) BackProp() {
	prevOut := l.prev.Prop("out").([]float64);
	inSize := l.prev.Prop("outSize").([]int)
	nextDeltas := l.next.Prop("deltas").([]float64);
	nextIn := l.next.Prop("in").([]float64);
	nextWeights := l.next.Prop("weights").([][]float64);

	for i:=0; i<l.size[0]; i++ {
		v := 0.0;

		for k:=0; k<len(nextIn); k++ {
			v -= nextDeltas[k]*DSigmoid(nextIn[k])*nextWeights[i][k];
		}

		l.deltas[i] = v;
	}

	for j:=0; j<l.size[0]; j++ {
		for i:=0; i<inSize[0];i++ {
			l.weights[i][j] += l.net.LearningRate()*l.deltas[j]*DSigmoid(l.in[j])*prevOut[i];
		}
	}
}

func (l *FCLayer) Prop(name string) interface{} {
	switch name {
	case "outSize": return l.size
	case "out": return l.out
	case "in": return l.in
	case "deltas": return l.deltas
	case "weights": return l.weights
	}

	return nil
}

func (l *FCLayer) State() interfaces.LayerState {
	out := make([]float64, len(l.out))
	copy(out, l.out)
	state := interfaces.LayerState{
		Class:l.class,
		Size: l.size,
		Out: out,
	}

	return state
}

func (l *FCLayer) WeightsState() interfaces.WeightsState {
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

func (l *FCLayer) SetWeightsState(weightsState interfaces.WeightsState) {
	l.weights = weightsState.Weights
}
