package config

import (
	"fmt"
	"os"
	"encoding/json"
	"io/ioutil"
	"path/filepath"
)

type Net struct {
	Size []int `json:"size"`
	Layers []Layer `json:"layers"`
}

type Layer struct {
	Class string `json:"type"`
	Size []int `json:"size"`
	Activate string `json:"activate"`
	/* Two below only for conv layer */
	Count int `json:"count"`
	Shape string `json:"shape"`
}

func GetNetConfig() Net {
	absPath, _ := filepath.Abs("convnet/config/net.json")
	file, e := ioutil.ReadFile(absPath)
	if e != nil {
		fmt.Printf("File error: %v\n", e)
		os.Exit(1)
	}
	net := Net{}

	json.Unmarshal(file, &net)

	return net
}
