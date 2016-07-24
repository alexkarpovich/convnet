package layers

import (
	//"fmt"
	"math"
	. "github.com/alexkarpovich/convnet/utils"
	"github.com/alexkarpovich/convnet/interfaces"
)

type PoolLayer struct {
	*Layer
	count int
	deltas []float64
}

func (l *PoolLayer) Prepare() {
	outSize := l.getOutSize()
	prevCount := l.prev.Prop("count").(int)
	prevInCount := l.prev.Prop("inCount").(int)
	l.count = prevCount * prevInCount
	l.out = make([]float64, l.count*outSize[0]*outSize[1])
	l.deltas = make([]float64, l.count*outSize[0]*outSize[1])
}

func (l *PoolLayer) FeedForward() {
	prevOut := l.prev.Prop("out").([]float64)
	inSize := l.prev.Prop("outSize").([]int)
	p := 0
	step := inSize[0]*inSize[1]

	for k:=0; k<l.count; k++ {
		offset := step * k

		for j:=0; j<inSize[1]; j+=l.size[1] {
			for i:=0; i<inSize[0]; i+=l.size[0] {
				highIndex := i+inSize[0]*j
				max := prevOut[highIndex+0+offset]

				for b:=0; b<l.size[1]; b++ {
					for a:=0; a<l.size[0]; a++ {
						max = math.Max(max, prevOut[highIndex+a+b*inSize[0]+offset]);
					}
				}

				//l.in[p] = max
				l.out[p] = max
				p++
			}
		}
	}
}

func (l *PoolLayer) BackProp() {
	nextClass := l.next.Class()
	nextDeltas := l.next.Prop("deltas").([]float64);
	nextIn := l.next.Prop("in").([]float64);
	outSize := l.getOutSize()
	l.deltas = make([]float64, l.count*outSize[0]*outSize[1])

	if nextClass == "conv" {
		nextKernels := l.next.Prop("kernels").([]float64)
		nextOutSize := l.next.Prop("outSize").([]int)
		nextCount := l.next.Prop("count").(int)
		nextKernelSize := l.next.Size()

		length := nextOutSize[0]*nextOutSize[1]
		kStep := nextKernelSize[0]*nextKernelSize[1]

		for z:=0; z<l.count; z++ {
			for k := 0; k < nextCount; k++ {
				kOffset := kStep * k
				data, preSize := PrepareInput(nextDeltas[k*length:k*length+length], "same", nextOutSize, nextKernelSize)

				reversedKernels := ReverseArray(nextKernels[kOffset:kOffset+kStep])
				_, A:= Conv2d(data, preSize, reversedKernels, nextKernelSize)


				for i:=0; i<length; i++ {
					l.deltas[i+z*length] += A[i]
				}
			}
		}


	} else {
		nextWeights := l.next.Prop("weights").([][]float64);

		for i := 0; i < len(l.out); i++ {
			v := 0.0;

			for k := 0; k < len(nextIn); k++ {
				v -= nextDeltas[k] * nextIn[k] * nextWeights[i][k];
			}

			l.deltas[i] = v;
		}
	}
}

func (l *PoolLayer) Prop(name string) interface{} {
	switch name {
	case "count": return l.count
	case "size": return l.size
	case "outSize": return l.getOutSize()
	case "out": return l.out
	case "deltas": return l.deltas
	}

	return nil
}

func (l *PoolLayer) getOutSize() []int {
	inSize := l.prev.Prop("outSize").([]int)

	return []int{inSize[0]/l.size[0], inSize[1]/l.size[1]}
}

func (l *PoolLayer) State() interfaces.LayerState {
	out := make([]float64, len(l.out))
	copy(out, l.out)
	state := interfaces.LayerState{
		Class:l.class,
		Size: l.size,
		Out: out,
		Count: l.count,
		OutSize: l.getOutSize(),
	}

	return state
}

func (l *PoolLayer) WeightsState() interfaces.WeightsState {
	weightsState := interfaces.WeightsState{
		Class: l.class,
	}
	return weightsState
}

func (l *PoolLayer) SetWeightsState(weightsState interfaces.WeightsState) {

}
