package utils

import(
	"math"
)

func Sigmoid(x float64) float64 {
	return 1/(1+math.Exp(-x))
}

func DSigmoid(x float64) float64 {
	y := Sigmoid(x)
	return y * (1 - y)
}

func PrepareInput(data []float64, shape string, imgSize []int, kernelSize []int) ([]float64, []int) {
	var leftTop, rightBottom int

	if shape == "valid" {
		return data, imgSize
	} else if shape == "same" {
		isEven := kernelSize[0]%2==0
		if isEven {
			leftTop = int(math.Floor(float64(kernelSize[0] / 2)));
			rightBottom = int(math.Floor(float64(leftTop / 2)));
		} else {
			leftTop = int(math.Floor(float64(kernelSize[0] / 2)))
			rightBottom = leftTop
		}
	} else if shape == "full" {
		leftTop = kernelSize[0]-1
		rightBottom = kernelSize[1]-1
	}

	outSize := make([]int, 2)

	outSize[0] = imgSize[0]+leftTop+rightBottom;
	outSize[1] = imgSize[1]+leftTop+rightBottom;

	result := make([]float64, outSize[0]*outSize[1])

	for j:=0; j<outSize[1];j++ {
		for i:=0; i<outSize[0]; i++ {
			if i<leftTop || i>imgSize[0]+leftTop-1 || j<leftTop || j>imgSize[1]+leftTop-1 {
				result[i+outSize[0]*j] = 0;
			} else {
				result[i+outSize[1]*j] = float64(data[i-leftTop+imgSize[0]*(j-leftTop)]);
			}
		}
	}

	return result, outSize
}

func Conv2d(input []float64, inputSize []int, kernel []float64, kernelSize []int) ([]float64, []float64) {
	p := 0
	iRange := inputSize[0]-kernelSize[0] + 1
	jRange := inputSize[1]-kernelSize[1] + 1
	length := iRange * jRange
	S := make([]float64, length)
	A := make([]float64, length)

	for j := 0; j < jRange; j++ {
		for i := 0; i < iRange; i++ {
			v := 0.0
			highIndex := i + inputSize[0] * j

			for b := 0; b < kernelSize[1]; b++ {
				for a := 0; a < kernelSize[0]; a++ {
					v += input[highIndex+a+b*inputSize[0]] * kernel[a+kernelSize[0]*b];
				}
			}

			S[p] = v
			A[p] = Sigmoid(v)
			p++
		}
	}

	return S, A
}

func ReverseArray(input []float64) []float64 {
	length := len(input)
	result := make([]float64, length)

	for i := range input {
		result[length-i-1] = input[i]
	}

	return result
}