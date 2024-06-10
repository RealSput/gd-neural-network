import '@g-js-api/g.js';

await $.exportConfig({
    type: "savefile",
    options: { info: true }
});

let wait_time = 0.02; // default: 0.1
let float_counter = (val = 0) => {
    let c_item = counter(0, false, false, true);
    $.add(object({
        OBJ_ID: 3614,
        START_TIME: val,
        STOP_TIME: 0,
        STOP_CHECKED: true,
        ITEM: c_item.item,
    }));
    c_item.display = (x, y, seconds_only = false) => $.add(object({
        OBJ_ID: 1615,
        X: x,
        Y: y,
        ITEM: c_item.item,
        TIME_COUNTER: true,
        SECONDS_ONLY: seconds_only,
        COLOR: color(1),
    }));
    c_item.add = (amount) => {
        if (typeof amount == 'number') {
            $.add(item_edit(undefined, undefined, c_item.item, TIMER, NONE, TIMER, ADD, undefined, undefined, amount));
        } else if (typeof amount == 'object') {
            $.add(item_edit(amount.item, undefined, c_item.item, amount.type, NONE, TIMER, ADD));
        }
    }
    c_item.subtract = (amount) => {
        if (typeof amount == 'number') {
            $.add(item_edit(undefined, undefined, c_item.item, TIMER, NONE, TIMER, SUB, undefined, undefined, amount));
        } else if (typeof amount == 'object') {
            $.add(item_edit(amount.item, undefined, c_item.item, amount.type, NONE, TIMER, SUB));
        }
    }
    c_item.divide = (amount) => {
        if (typeof amount == 'number') {
            $.add(item_edit(undefined, undefined, c_item.item, TIMER, NONE, TIMER, DIV, undefined, undefined, amount));
        } else if (typeof amount == 'object') {
            $.add(item_edit(amount.item, undefined, c_item.item, amount.type, NONE, TIMER, DIV));
        }
    }
    c_item.multiply = (amount) => {
        if (typeof amount == 'number') {
            $.add(item_edit(undefined, undefined, c_item.item, TIMER, NONE, TIMER, MUL, undefined, undefined, amount));
        } else if (typeof amount == 'object') {
            $.add(item_edit(amount.item, undefined, c_item.item, amount.type, NONE, TIMER, MUL));
        }
    }
    c_item.set = (amount) => {
        if (typeof amount == 'number') {
            $.add(object({
                OBJ_ID: 3614,
                START_TIME: amount,
                STOP_TIME: 0,
                STOP_CHECKED: true,
                ITEM: c_item.item,
            }));
        } else if (typeof amount == 'object') {
            $.add(item_edit(amount.item, undefined, c_item.item, amount.type, NONE, TIMER, EQ));
        }
    }
    return c_item;
}
class NeuralNetwork {
    constructor(inputNeurons, hiddenNeurons, outputNeurons) {
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.outputNeurons = outputNeurons;
        this.inputNeuronCounters = Array(inputNeurons).fill(0).map(() => float_counter());
        this.hiddenNeuronCounters = Array(hiddenNeurons).fill(0).map(() => float_counter());
        this.outputOutputCounters = Array(outputNeurons).fill(0).map(() => float_counter());

        this.weightsInputHidden = this.initializeWeights(inputNeurons, hiddenNeurons);
        this.weightsHiddenOutput = this.initializeWeights(hiddenNeurons, outputNeurons);

        this.weightsInputHiddenCounters = this.weightsInputHidden.map(x => x.map(y => float_counter(y)));
        this.weightsHiddenOutputCounters = this.weightsHiddenOutput.map(x => x.map(y => float_counter(y)));
        console.log(this.weightsHiddenOutputCounters);

        this.biasHidden = new Array(hiddenNeurons).fill(1);
        this.biasOutput = new Array(outputNeurons).fill(1);

        this.biasHiddenCounters = this.biasHidden.map(x => float_counter(x));
        this.biasOutputCounters = this.biasOutput.map(x => float_counter(x));

        this.inputs = new Array(inputNeurons).fill(0);

        this.hiddenOutputs = new Array(hiddenNeurons).fill(0);
        this.hiddenOutputCounters = new Array(hiddenNeurons).fill(0).map(() => float_counter());

        this.outputOutputs = new Array(outputNeurons).fill(0);
        this.outputOutputCounters = new Array(outputNeurons).fill(0).map(() => float_counter());

        this.outputGradients = new Array(outputNeurons).fill(0);
        this.outputGradientCounters = this.outputGradients.map(x => float_counter(x));

        this.hiddenGradients = new Array(hiddenNeurons).fill(0);
        this.hiddenGradientCounters = this.hiddenGradients.map(x => float_counter(x));
    }

    initializeWeights(inputNeurons, outputNeurons) {
        const weights = new Array(inputNeurons);
        for (let i = 0; i < inputNeurons; i++) {
            weights[i] = new Array(outputNeurons);
            for (let j = 0; j < outputNeurons; j++) {
                weights[i][j] = Math.random() * 0.2 - 0.1;
            }
        }
        return weights;
    }

    setInputData(data) {
        if (data.length === this.inputNeurons) {
            this.inputs = data;
        } else {
            console.error("Input data length does not match the number of input neurons.");
        }
    }

    GD_setInputData(data) {
        if (data.length === this.inputNeurons) {
            this.inputNeuronCounters.forEach((x, i) => x.set(data[i]))
        } else {
            console.error("Input data length does not match the number of input neurons.");
        }
    }

    setTrainingData(data) {
        if (data.length > 0 && data[0].input.length === this.inputNeurons && data[0].output.length === this.outputNeurons) {
            this.trainingData = data;
        } else {
            console.error("Invalid training data format or size.");
        } 
    }

    passThroughHiddenLayer() {
        for (let i = 0; i < this.hiddenNeurons; i++) {
            let sum = 0;
            for (let j = 0; j < this.inputNeurons; j++) {
                sum += this.inputs[j] * this.weightsInputHidden[j][i];
            }
            sum += this.biasHidden[i];
            this.hiddenOutputs[i] = Math.max(0, sum);
        }
    }

    GD_passThroughHiddenLayer() {
        for (let i = 0; i < this.hiddenNeurons; i++) {
            for (let j = 0; j < this.inputNeurons; j++) {
                $.add(item_edit(this.inputNeuronCounters[j].item, this.weightsInputHiddenCounters[j][i].item, this.hiddenOutputCounters[i].item, TIMER, TIMER, TIMER, ADD, MUL, NONE).with(obj_props.X, 75).with(obj_props.Y, 145));
                wait(wait_time);
            }
            this.hiddenOutputCounters[i].add(this.biasHiddenCounters[i]);
            let relu_cond = trigger_function(() => {
                this.hiddenOutputCounters[i].set(0);
            });
            $.add(item_comp(this.hiddenOutputCounters[i].item, 0, TIMER, NONE, LESS, relu_cond, undefined, 1, 0));
        }
    }

    passThroughOutputLayer() {
        for (let i = 0; i < this.outputNeurons; i++) {
            let sum = 0;
            for (let j = 0; j < this.hiddenNeurons; j++) {
                sum += this.hiddenOutputs[j] * this.weightsHiddenOutput[j][i];
            }
            sum += this.biasOutput[i];
            this.outputOutputs[i] = sum;
        }
    }

    GD_passThroughOutputLayer() {
        for (let i = 0; i < this.outputNeurons; i++) {
            for (let j = 0; j < this.hiddenNeurons; j++) {
                $.add(item_edit(this.hiddenOutputCounters[j].item, this.weightsHiddenOutputCounters[j][i].item, this.outputOutputCounters[i].item, TIMER, TIMER, TIMER, ADD, MUL, NONE).with(obj_props.X, 145).with(obj_props.Y, 145));
                wait(wait_time)
            }
            this.outputOutputCounters[i].add(this.biasOutputCounters[i]);
            $.add(item_edit(undefined, undefined, this.outputOutputCounters[i].item, NONE, NONE, TIMER, MUL, NONE, NONE, 1, NONE, NONE, NONE, RND).with(obj_props.X, 115).with(obj_props.Y, 145)); // rounds this.outputOutputCounters[i]
        }
    }

    feedForward(inputData) {
        this.setInputData(inputData);
        this.passThroughHiddenLayer();
        this.passThroughOutputLayer();
    }

    GD_feedForward() {
        this.GD_passThroughHiddenLayer();
        this.GD_passThroughOutputLayer();
    }

    calculateError(target) {
        let totalError = 0;
        for (let i = 0; i < this.outputNeurons; i++) {
            totalError += Math.pow((target[i] - this.outputOutputs[i]), 2);
        }
        return totalError / this.outputNeurons;
    }

    passErrorToOutputLayer(target) {
        for (let i = 0; i < this.outputNeurons; i++) {
            const error = target[i] - this.outputOutputs[i];
            this.outputGradients[i] = error;
        }
    }

    GD_passErrorToOutputLayer(target) {
        for (let i = 0; i < this.outputNeurons; i++) {
            const error = target[i] - this.outputOutputs[i];
            this.outputGradientCounters[i].set(error);
        }
    }

    passErrorToHiddenLayer() {
        for (let i = 0; i < this.hiddenNeurons; i++) {
            let errorGradient = 0;
            for (let j = 0; j < this.outputNeurons; j++) {
                errorGradient += this.outputGradients[j] * this.weightsHiddenOutput[i][j];
            }
            this.hiddenGradients[i] = errorGradient * (this.hiddenOutputs[i] > 0 ? 1 : 0);
        }
    }

    GD_passErrorToHiddenLayer() {
        for (let i = 0; i < this.hiddenNeurons; i++) {
            let errorGradient = float_counter();
            for (let j = 0; j < this.outputNeurons; j++) {
                // errorGradient += this.outputGradients[j] * this.weightsHiddenOutput[i][j];
                $.add(item_edit(this.outputGradientCounters[j].item, this.weightsHiddenOutputCounters[i][j].item, errorGradient.item, TIMER, TIMER, TIMER, ADD, MUL, NONE))
                wait(wait_time);
            }
            this.hiddenOutputCounters[i].if_is(LARGER_THAN, 0, trigger_function(() => {
                this.hiddenOutputCounters[i].set(1);
            }))
            this.hiddenOutputCounters[i].if_is(SMALLER_THAN, 0, trigger_function(() => {
                this.hiddenOutputCounters[i].set(0);
            }))
            wait(wait_time);
            errorGradient.multiply(this.hiddenOutputCounters[i]);
            this.hiddenGradientCounters[i].set(errorGradient)
        }
    }    

    updateWeights(learningRate = 0.001) {
        const clipValue = 1.0;

        for (let i = 0; i < this.inputNeurons; i++) {
            for (let j = 0; j < this.hiddenNeurons; j++) {
                let gradient = this.hiddenGradients[j];
                if (gradient > clipValue) gradient = clipValue;
                if (gradient < -clipValue) gradient = -clipValue;

                this.weightsInputHidden[i][j] += learningRate * this.inputs[i] * gradient;
            }
        }

        for (let i = 0; i < this.hiddenNeurons; i++) {
            for (let j = 0; j < this.outputNeurons; j++) {
                let gradient = this.outputGradients[j];
                if (gradient > clipValue) gradient = clipValue;
                if (gradient < -clipValue) gradient = -clipValue;

                this.weightsHiddenOutput[i][j] += learningRate * this.hiddenOutputs[i] * gradient;
            }
        }

        for (let i = 0; i < this.hiddenNeurons; i++) {
            let gradient = this.hiddenGradients[i];
            if (gradient > clipValue) gradient = clipValue;
            if (gradient < -clipValue) gradient = -clipValue;

            this.biasHidden[i] += learningRate * gradient;
        }

        for (let i = 0; i < this.outputNeurons; i++) {
            let gradient = this.outputGradients[i];
            if (gradient > clipValue) gradient = clipValue;
            if (gradient < -clipValue) gradient = -clipValue;

            this.biasOutput[i] += learningRate * gradient;
        }
    }

    GD_updateWeights(learningRate = 0.001) {
        const clipValue = 1.0;

        for (let i = 0; i < this.inputNeurons; i++) {
            for (let j = 0; j < this.hiddenNeurons; j++) {
                let gradient = this.hiddenGradientCounters[j];
                gradient.if_is(LARGER_THAN, clipValue, trigger_function(() => gradient.set(clipValue)));
                gradient.if_is(SMALLER_THAN, clipValue, trigger_function(() => gradient.set(-clipValue)));
                // if (gradient > clipValue) gradient = clipValue;
                // if (gradient < -clipValue) gradient = -clipValue;

                // this.weightsInputHidden[i][j] += learningRate * this.inputs[i] * gradient;
                
                $.add(item_edit(this.inputNeuronCounters[i].item, gradient.item, this.weightsInputHiddenCounters[i][j].item, TIMER, TIMER, TIMER, ADD, MUL, MUL, learningRate));
                wait(wait_time);
            }
        }

        for (let i = 0; i < this.hiddenNeurons; i++) {
            for (let j = 0; j < this.outputNeurons; j++) {
                let gradient = this.outputGradientCounters[j];
                gradient.if_is(LARGER_THAN, clipValue, trigger_function(() => gradient.set(clipValue)));
                gradient.if_is(SMALLER_THAN, clipValue, trigger_function(() => gradient.set(-clipValue)));

                // this.weightsHiddenOutput[i][j] += learningRate * this.hiddenOutputs[i] * gradient;
                $.add(item_edit(this.hiddenOutputCounters[i].item, gradient.item, this.weightsHiddenOutputCounters[i][j].item, TIMER, TIMER, TIMER, ADD, MUL, MUL, learningRate));
                wait(wait_time);
            }
        }

        for (let i = 0; i < this.hiddenNeurons; i++) {
            let gradient = this.hiddenGradientCounters[i];
            gradient.if_is(LARGER_THAN, clipValue, trigger_function(() => gradient.set(clipValue)));
            gradient.if_is(SMALLER_THAN, clipValue, trigger_function(() => gradient.set(-clipValue)));

            // this.biasHidden[i] += learningRate * gradient;
            $.add(item_edit(gradient.item, undefined, this.biasHiddenCounters[i].item, TIMER, NONE, TIMER, ADD, MUL, undefined, learningRate));
            wait(wait_time);
        }

        for (let i = 0; i < this.outputNeurons; i++) {
            let gradient = this.outputGradientCounters[i];
            gradient.if_is(LARGER_THAN, clipValue, trigger_function(() => gradient.set(clipValue)));
            gradient.if_is(SMALLER_THAN, clipValue, trigger_function(() => gradient.set(-clipValue)));

            // this.biasOutput[i] += learningRate * gradient;
            $.add(item_edit(gradient.item, undefined, this.biasOutputCounters[i].item, TIMER, NONE, TIMER, ADD, MUL, undefined, learningRate));
            wait(wait_time);
        }
    }
}

const neuralNetwork = new NeuralNetwork(3, 20, 1);

const trainingData = [
    { input: [1, 2, 3], output: [4] },
    { input: [1, 2, 4], output: [8] },
    { input: [5, 10, 15], output: [20] },
    { input: [1, 1, 2], output: [3] },
    { input: [2, 4, 6], output: [8] },
    { input: [3, 6, 9], output: [12] },
    { input: [10, 20, 30], output: [40] },
    { input: [2, 3, 5], output: [8] },
    { input: [4, 8, 12], output: [16] },
    { input: [7, 14, 21], output: [28] },
    { input: [5, 5, 10], output: [15] },
    { input: [2, 2, 4], output: [6] },
    { input: [3, 3, 6], output: [9] },
    { input: [11, 22, 33], output: [44] },
    { input: [6, 12, 18], output: [24] },
    { input: [13, 26, 39], output: [52] },
    { input: [4, 7, 11], output: [15] },
    { input: [8, 16, 24], output: [32] },
    { input: [15, 30, 45], output: [60] },
    { input: [9, 18, 27], output: [36] },
    { input: [21, 42, 63], output: [84] },
    { input: [8, 8, 16], output: [24] },
    { input: [10, 20, 30], output: [40] },
    { input: [12, 24, 36], output: [48] },
    { input: [14, 28, 42], output: [56] },
    { input: [16, 32, 48], output: [64] },
    { input: [18, 36, 54], output: [72] },
    { input: [20, 40, 60], output: [80] },
    { input: [22, 44, 66], output: [88] },
    { input: [24, 48, 72], output: [96] },
    { input: [26, 52, 78], output: [104] },
    { input: [28, 56, 84], output: [112] },
    { input: [30, 60, 90], output: [120] },
    { input: [32, 64, 96], output: [128] },
    { input: [34, 68, 102], output: [136] },
    { input: [36, 72, 108], output: [144] },
    { input: [38, 76, 114], output: [152] },
    { input: [40, 80, 120], output: [160] },
    { input: [42, 84, 126], output: [168] },
    { input: [44, 88, 132], output: [176] },
    { input: [46, 92, 138], output: [184] },
    { input: [48, 96, 144], output: [192] },
    { input: [50, 100, 150], output: [200] },
    { input: [52, 104, 156], output: [208] },
    { input: [54, 108, 162], output: [216] },
    { input: [56, 112, 168], output: [224] },
    { input: [58, 116, 174], output: [232] },
    { input: [60, 120, 180], output: [240] },
    { input: [62, 124, 186], output: [248] },
    { input: [64, 128, 192], output: [256] },
    { input: [66, 132, 198], output: [264] },
    { input: [68, 136, 204], output: [272] },
    { input: [70, 140, 210], output: [280] },
];

const iterations = 10000;
// todo: port training to GD

for (let i = 0; i < iterations; i++) {
    for (const data of trainingData) {
        neuralNetwork.feedForward(data.input);
        const error = neuralNetwork.calculateError(data.output);
        neuralNetwork.passErrorToOutputLayer(data.output);
        neuralNetwork.passErrorToHiddenLayer();
        neuralNetwork.updateWeights(0.001);
    }
}

let seq_arr = trainingData.map(x => {
    return [ trigger_function(() => {
        neuralNetwork.GD_setInputData(x.input);
        neuralNetwork.GD_feedForward();
        neuralNetwork.GD_passErrorToOutputLayer(x.output);
    }), 1];
});
let cycleTrainingData = sequence(seq_arr, 1);
let epochs = counter(0);
let train_iter = counter();
train_iter.display(45, 15);
epochs.display(75, 15);
let reset_all = trigger_function(() => {
    neuralNetwork.inputNeuronCounters.forEach((x) => x.set(0));
    neuralNetwork.hiddenNeuronCounters.forEach((x) => x.set(0));
    neuralNetwork.hiddenNeuronCounters.forEach((x) => x.set(0)); 
    neuralNetwork.weightsInputHiddenCounters.forEach(x => x.forEach(y => y.set(0)));
    neuralNetwork.weightsHiddenOutputCounters.forEach(x => x.forEach(y => y.set(0)));
    neuralNetwork.biasHiddenCounters.forEach(x => x.set(0));
    neuralNetwork.biasOutputCounters.forEach(x => x.set(0));
    neuralNetwork.hiddenOutputCounters.forEach(x => x.set(0));
    neuralNetwork.outputOutputCounters.forEach(x => x.set(0));
    neuralNetwork.outputGradientCounters.forEach(x => x.set(0));
    neuralNetwork.hiddenGradientCounters.forEach(x => x.set(0));
});
for_loop(range(0, iterations), () => {
    reset_all.call();
    wait(wait_time);
    cycleTrainingData();
    wait(wait_time * 5);
    neuralNetwork.GD_passErrorToHiddenLayer(); // 2
    neuralNetwork.GD_updateWeights(0.001);
    train_iter.add(1);
    train_iter.if_is(EQUAL_TO, trainingData.length, trigger_function(() => {
        train_iter.reset();
        epochs.add(1)
    }));
});

neuralNetwork.inputNeuronCounters.forEach((x, i) => x.display(i * 70 + 45, 105))
neuralNetwork.hiddenOutputCounters.forEach((x, i) => x.display(i * 70 + 45, 135))
neuralNetwork.outputOutputCounters.forEach((x, i) => x.display(i * 70 + 45, 75))

wait(1);
neuralNetwork.feedForward([1, 2, 3]);
console.log(neuralNetwork.outputOutputs);
let run = trigger_function(() => {
    neuralNetwork.GD_feedForward();
});
console.log('group to call:', run.value);