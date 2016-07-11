import {Utils} from './utils'
import {Layer, LayerType} from './Layer'
import {ConvLayer} from './layers/ConvLayer'
import {PoolingLayer} from './layers/PoolingLayer'
import {FCLayer} from './layers/FCLayer'
import {OutputLayer} from './layers/OutputLayer'

export interface IConvnetParams {
    size:number[];
    layers:any[];
}

export class Convnet {
    private in:number[];
    private out:number[];
    private label:number[];
    private size:number[];
    private layers:any[];

    constructor(params:IConvnetParams) {
        this.size = params.size;
        this.prepareLayers(params.layers);
    }

    private prepareInput(image) {
        this.in = Utils.img2data(image);
    }

    private prepareLayers(layers:any) {
        this.layers = [];
        let prevLayer = null;

        layers.forEach(layer => {
            let currentLayer;
            layer['prev'] = prevLayer;
            layer['net'] = this;

            switch(layer.type) {
                case 'conv':
                    currentLayer = new ConvLayer(layer);
                    break;
                case 'pooling':
                    currentLayer = new PoolingLayer(layer);
                    break;
                case 'fc':
                    currentLayer = new FCLayer(layer);
                    break;
                case 'output':
                    currentLayer = new OutputLayer(layer);
                    break;
            }

            this.layers.push(currentLayer);
            prevLayer && prevLayer.setNextLayer(currentLayer);
            prevLayer = currentLayer;
        });
    }

    public train(params:any, callback) {
        console.info('Convnet:train');
        let i = 1, error = 0;
        while (i <= params.maxIterations || error < params.minError) {
            error = 0;
            console.group('Iteration '+i);
            console.time(''+i);
            params.trainingSet.forEach(example => {
                this.setExample(example.input, example.label);
                this.feadforward();
                error += this.backprop();
            });
            console.info('Error:', error);
            console.timeEnd(''+i);
            console.groupEnd();
            i++;
        }
        callback && callback();
    }

    public test(image:any) {
        console.info('Convnet:test');
        this.setInput(image);
        this.feadforward();
        console.log(this.out);

        return this.out;
    }

    public feadforward() {
        this.layers.forEach(layer => layer.feadforward());
    }

    public backprop() {
        let error = 0;
        let inversedLayers = this.layers.reverse();
        inversedLayers.forEach(layer => {
            layer.backprop();
            if (layer.getType()==LayerType.OUTPUT) {
                error+=layer.getError();
            }
        });

        return error;
    }

    public getInput() {
        return this.in;
    }

    public getLabel() {
        return this.label;
    }

    public getSize() {
        return this.size;
    }

    public setInput(image:ImageData) {
        this.prepareInput(image);
        this.size = [image.width, image.height];
    }

    public setOutput(out:number[]) {
        this.out = out;
    }

    public setExample(input:any, label:any) {
        this.prepareInput(input);
        this.label = label;
    }
}
