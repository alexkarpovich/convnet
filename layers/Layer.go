package layers

import (
	"github.com/alexkarpovich/convnet/interfaces"
)

type ILayer interface {
	Init(interfaces.INet, string, []int)
	SetPrev(ILayer)
	SetNext(ILayer)
	Prepare()
	Class() string
	Size() []int
	Prop(string) interface{}
	FeedForward()
	BackProp()
	State() interfaces.LayerState
	WeightsState() interfaces.WeightsState
	SetWeightsState(interfaces.WeightsState)
}

type Layer struct {
	net interfaces.INet
	class string
	prev ILayer
	next ILayer
	size []int
	in []float64
	out []float64
}

func (l *Layer) Init(net interfaces.INet, class string, size []int) {
	l.net = net
	l.class = class
	l.size = size
}

func (l *Layer) SetPrev(prevLayer ILayer) {
	l.prev = prevLayer
}

func (l *Layer) SetNext(nextLayer ILayer) {
	l.next = nextLayer
}

func (l *Layer) Class() string {
	return l.class
}

func (l *Layer) Size() []int {
	return l.size
}
