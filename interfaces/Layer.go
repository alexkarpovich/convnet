package interfaces

type LayerState struct {
	Class string `json:"type"`
	Size []int `json:"size"`
	Out []float64 `json:"out"`
	InCount int `json:"inCount"`
	OutSize []int `json:"outSize"`
	Count int `json:"count"`
}

type WeightsState struct {
	Class string `json:"type"`
	Weights [][]float64 `json:"weights"`
	Kernels []float64 `json:"kernels"`
}
