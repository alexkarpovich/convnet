package convnet

import (
	"fmt"
	"time"
	"image"
	"runtime"
	"github.com/petar/GoMNIST"
	"github.com/alexkarpovich/convnet/config"
	. "github.com/alexkarpovich/convnet/layers"
	. "github.com/alexkarpovich/convnet/interfaces"
)

type Net struct {
	size []int
	in []float64
	out []float64
	layers []ILayer
	label []float64
	err float64
}

func (net *Net) FromConfig(netConfig config.Net) {
	net.size = netConfig.Size
	net.initLayers(netConfig.Layers)
}

func (net *Net) Init() {
	netConfig := config.GetNetConfig()
	net.size = netConfig.Size
	net.initLayers(netConfig.Layers)
}

func (net *Net) initLayers(layersConfig []config.Layer) {
	net.layers = make([]ILayer, len(layersConfig))
	var currentLayer, prevLayer ILayer

	for i := range layersConfig {
		lconf := layersConfig[i]
		layer := &Layer{}
		layer.Init(net, lconf.Class, lconf.Size)

		switch lconf.Class {
		case "conv":
			convLayer := &ConvLayer{Layer:layer}
			convLayer.Construct(lconf.Shape, lconf.Count)
			currentLayer = convLayer
			break
		case "pool": currentLayer = &PoolLayer{Layer:layer}
			break
		case "fc": currentLayer = &FCLayer{Layer:layer}
			break
		case "output": currentLayer = &OutputLayer{Layer:layer}
		}

		if i==0 {
			currentLayer.SetPrev(nil)
		} else {
			currentLayer.SetPrev(prevLayer)
			prevLayer.SetNext(currentLayer)
		}

		net.layers[i] = currentLayer
		prevLayer = currentLayer
	}

	for i := range net.layers {
		net.layers[i].Prepare()
	}
}

func (net *Net) Train(params TrainParams, trainingSet *GoMNIST.Set) {
	var err float64 = 10.0
	var mem runtime.MemStats
	iter := 1

	for err > params.MinError {
		err = 0
		start := time.Now()

		for i:=0; i<100; i++ {
			net.setMNISTExample(trainingSet.Images[i], trainingSet.Labels[i])
			net.forward()
			net.backward()

			err += net.err
		}

		elapsed := time.Since(start)

		runtime.ReadMemStats(&mem)
		fmt.Println(iter, "E =", err, ", took=", elapsed, "sec, alloc=", mem.Alloc/1048576, "MB")

		iter++
	}
}

func (net *Net) Test(img image.Image) {
	bounds := img.Bounds()
	net.size = []int{bounds.Max.X, bounds.Max.Y}
	net.prepareInput(img)
	net.forward()

	fmt.Println(net.out)
}

func (net *Net) GetSize() []int {
	return net.size
}

func (net *Net) GetInput() []float64 {
	return net.in
}

func (net *Net) GetLabel() []float64 {
	return net.label
}

func (net *Net) SetOutput(output []float64) {
	net.out = output
}

func (net *Net) SetError(err float64) {
	net.err = err
}

func (net *Net) prepareInput(img image.Image) {
	net.in = make([]float64, net.size[0] * net.size[1])

	for j:=0; j<net.size[1]; j++ {
		for i:=0; i<net.size[0]; i++ {
			r, g, b, _ := img.At(i, j).RGBA()
			net.in[i+net.size[0]*j] = 0.2989*float64(r) + 0.5870*float64(g) + 0.1140*float64(b);
		}
	}
}

func (net *Net) forward() {
	for i := range net.layers {
		net.layers[i].FeedForward()
	}
}

func (net *Net) backward() {
	for i := range net.layers {
		net.layers[len(net.layers)-i-1].BackProp()
	}
}

func (net *Net) setExample(img image.Image, label []float64) {
	net.prepareInput(img)
	net.label = label
}

func (net *Net) setMNISTExample(img GoMNIST.RawImage, label GoMNIST.Label) {
	net.prepareMNISTInput(img)
	net.prepareMNISTLabel(label)
}

func (net *Net) prepareMNISTInput(img GoMNIST.RawImage) {
	net.in = make([]float64, net.size[0] * net.size[1])

	for i:=0; i<len(img); i++ {
		net.in[i] = float64(img[i]);
	}
}

func (net *Net) prepareMNISTLabel(label GoMNIST.Label) {
	net.label = make([]float64, 10)

	for i:=0; i<10; i++ {
		if int(label) == i {
			net.label[i] = 1
		} else {
			net.label[i] = 0;
		}
	}
}