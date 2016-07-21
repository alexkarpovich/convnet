package layers

import (
	//"fmt"
	"math/rand"
	. "github.com/alexkarpovich/convnet/api/convnet/utils"
)

type FCLayer struct {
	*Layer
	weights [][]float64
	deltas []float64
}

func (l *FCLayer) Prepare() {
	inSize := l.prev.GetProp("outSize").([]int)
	count := l.prev.GetProp("count").(int)
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
	prevOut := l.prev.GetProp("out").([]float64)
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
	alpha := 0.001;
	prevOut := l.prev.GetProp("out").([]float64);
	inSize := l.prev.GetProp("outSize").([]int)
	nextDeltas := l.next.GetProp("deltas").([]float64);
	nextIn := l.next.GetProp("in").([]float64);
	nextWeights := l.next.GetProp("weights").([][]float64);

	for i:=0; i<l.size[0]; i++ {
		v := 0.0;

		for k:=0; k<len(nextIn); k++ {
			v -= nextDeltas[k]*DSigmoid(nextIn[k])*nextWeights[i][k];
		}

		l.deltas[i] = v;
	}

	for j:=0; j<l.size[0]; j++ {
		for i:=0; i<inSize[0];i++ {
			l.weights[i][j] += alpha*l.deltas[j]*DSigmoid(l.in[j])*prevOut[i];
		}
	}
}

func (l *FCLayer) GetProp(name string) interface{} {
	switch name {
	case "outSize": return l.size
	case "out": return l.out
	case "in": return l.in
	case "deltas": return l.deltas
	case "weights": return l.weights
	}

	return nil
}
