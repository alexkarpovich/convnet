package layers

import (
	//"fmt"
	"math/rand"
	. "github.com/alexkarpovich/convnet/utils"
)

type ConvLayer struct {
	*Layer
	shape string 		// VALID, SAME, FULL
	count int			// Count of feature maps
	inCount int			// Count of inputs (when prev layer for eg Pooling)
	imSize []int		// Prepared image size for appropriate shape
	kernels []float64
	im []float64		// Prepared image data for appropriate shape
	deltas []float64
}

func (l *ConvLayer) Construct(shape string, count int) {
	l.shape = shape
	l.count = count
	l.inCount = 1
}

func (l *ConvLayer) Prepare() {
	if l.prev != nil {
		prevClass := l.prev.GetClass()
		if prevClass=="conv" || prevClass=="pool" {
			l.inCount = l.prev.GetProp("count").(int)
		}
	}

	outSize := l.getOutSize()
	klength := l.count*l.size[0]*l.size[1]
	length := outSize[0]*outSize[1]*l.count

	l.imSize = make([]int, 2)
	l.kernels = make([]float64, klength)
	l.im = make([]float64, outSize[0]*outSize[1]*l.inCount)
	l.in = make([]float64, length*l.inCount)
	l.out = make([]float64, length*l.inCount)
	l.deltas = make([]float64, length*l.inCount)

	for i := 0; i < klength; i++ {
		l.kernels[i] = rand.Float64()*2-1
	}


}

func (l *ConvLayer) GetProp(name string) interface{} {
	switch name {
	case "count": return l.count
	case "inCount": return l.inCount
	case "kernelSize": return l.size
	case "outSize": return l.getOutSize()
	case "in": return l.in
	case "out": return l.out
	case "deltas": return l.deltas
	case "kernels": return l.kernels
	}

	return nil
}

func (l *ConvLayer) FeedForward() {
	isFirst := true
	input := l.net.GetInput()
	inputSize := l.net.GetSize()
	var prevOut []float64

	if l.prev != nil {
		prevClass := l.prev.GetClass()
		isFirst = false

		if prevClass=="conv" || prevClass=="pool" {
			l.inCount = l.prev.GetProp("count").(int)
			inputSize = l.prev.GetProp("outSize").([]int)
			prevOut = l.prev.GetProp("out").([]float64)
		}
	}

	l.im = []float64{}
	l.in = []float64{}
	l.out = []float64{}
	length := inputSize[0]*inputSize[1]
	kStep := l.size[0] * l.size[1]

	for z:=0; z<l.inCount; z++ {
		if isFirst == false {
			input = prevOut[z*length:z*length+length]
		}

		data, outSize := PrepareInput(input, l.shape, inputSize, l.size)

		for k := 0; k < l.count; k++ {
			kOffset := kStep * k

			S, A := Conv2d(data, outSize, l.kernels[kOffset:kOffset+kStep], l.size)
			l.in = append(l.in, S...)
			l.out = append(l.out, A...)
		}

		l.im = append(l.im, data...)
		l.imSize = outSize
	}
}

func (l *ConvLayer) BackProp() {
	alpha := 0.001;
	nextDeltas := l.next.GetProp("deltas").([]float64);
	subsize := l.next.GetProp("size").([]int);
	convSize := l.getOutSize();
	sLength := subsize[0]*subsize[1]
	cLength := convSize[0]*convSize[1]
	iRange := l.imSize[0]-l.size[0];
	jRange := l.imSize[1]-l.size[1];
	t, p := 0, 0


	for z:=0; z<l.inCount; z++ {
		l.computeDeltas(z, cLength, sLength, convSize, subsize, nextDeltas, &p)
		l.correctWeights(z, sLength, iRange, jRange, alpha, &t)
	}
}

func (l *ConvLayer) computeDeltas(z, cLength, sLength int, convSize, subsize []int, nextDeltas []float64, p *int) {
	for k := 0; k < l.count; k++ {
		offset := cLength*(k+z);
		for j := 0; j < convSize[1]; j += subsize[1] {
			for i := 0; i < convSize[0]; i += subsize[0] {
				delta := nextDeltas[*p] / float64(sLength)
				highIndex := i+convSize[0]*j

				for b := 0; b < subsize[1]; b++ {
					for a := 0; a < subsize[0]; a++ {
						l.deltas[highIndex+a+b*convSize[0]+offset] = delta;
					}
				}
				(*p)++
			}
		}
	}
}

func (l *ConvLayer) correctWeights(z, sLength, iRange, jRange int, alpha float64, t *int) {
	for k:=0; k<l.count; k++ {
		offset := k*sLength

		for j:=0; j<jRange; j++ {
			for i:=0; i<iRange; i++ {
				highIndex := i+l.imSize[0]*j+z*l.imSize[0]*l.imSize[1];

				for b:=0; b<l.size[1]; b++ {
					for a:=0; a<l.size[0]; a++ {
						l.kernels[a+l.size[0]*b+offset] +=
							alpha*l.deltas[*t]*DSigmoid(l.in[*t])*l.im[highIndex+a+b*l.imSize[0]];
					}
				}

				(*t)++
			}
		}
	}
}

func (l *ConvLayer) getOutSize() []int {
	inSize := l.net.GetSize()

	if l.prev != nil {
		inSize = l.prev.GetProp("outSize").([]int)
	}

	switch l.shape {
	case "valid": return []int{inSize[0]-l.size[0]+1, inSize[1]-l.size[1]+1}
	case "same": return []int{inSize[0], inSize[1]}
	case "full": return []int{inSize[0]+l.size[0]-1, inSize[1]+l.size[1]-1}
	}

	return inSize
}