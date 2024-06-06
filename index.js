import '@g-js-api/g.js';

await $.exportConfig({
    type: "savefile",
    options: { info: true }
});

let float_counter_arg;
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
            $.add(object({
                OBJ_ID: 3614,
                START_TIME: amount,
                STOP_TIME: 0,
                STOP_CHECKED: true,
                ITEM: float_counter_arg.item,
            }));
            $.add(item_edit(float_counter_arg.item, undefined, c_item.item, TIMER, NONE, TIMER, ADD));
        } else if (typeof amount == 'object') {
            $.add(item_edit(amount.item, undefined, c_item.item, amount.type, NONE, TIMER, ADD));
        }
    }
    c_item.subtract = (amount) => {
        if (typeof amount == 'number') {
            $.add(object({
                OBJ_ID: 3614,
                START_TIME: amount,
                STOP_TIME: 0,
                STOP_CHECKED: true,
                ITEM: float_counter_arg.item,
            }));
            $.add(item_edit(float_counter_arg.item, undefined, c_item.item, TIMER, NONE, TIMER, SUB));
        } else if (typeof amount == 'object') {
            $.add(item_edit(amount.item, undefined, c_item.item, amount.type, NONE, TIMER, SUB));
        }
    }
    c_item.divide = (amount) => {
        if (typeof amount == 'number') {
            $.add(object({
                OBJ_ID: 3614,
                START_TIME: amount,
                STOP_TIME: 0,
                STOP_CHECKED: true,
                ITEM: float_counter_arg.item,
            }));
            $.add(item_edit(float_counter_arg.item, undefined, c_item.item, TIMER, NONE, TIMER, DIV));
        } else if (typeof amount == 'object') {
            $.add(item_edit(amount.item, undefined, c_item.item, amount.type, NONE, TIMER, DIV));
        }
    }
    c_item.multiply = (amount) => {
        if (typeof amount == 'number') {
            $.add(object({
                OBJ_ID: 3614,
                START_TIME: amount,
                STOP_TIME: 0,
                STOP_CHECKED: true,
                ITEM: float_counter_arg.item,
            }));
            $.add(item_edit(float_counter_arg.item, undefined, c_item.item, TIMER, NONE, TIMER, MUL));
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
float_counter_arg = float_counter();
class NeuralNetwork {
    constructor(inputNeurons, hiddenNeurons, outputNeurons) {
        this.inputNeurons = inputNeurons;
        this.hiddenNeurons = hiddenNeurons;
        this.outputNeurons = outputNeurons;
        this.inputNeuronCounters = Array(inputNeurons).fill(0).map(() => float_counter());
        this.hiddenNeuronCounters = Array(hiddenNeurons).fill(0).map(() => float_counter());
        this.outputNeuronCounters = Array(outputNeurons).fill(0).map(() => float_counter());

        this.weightsInputHidden = this.initializeWeights(inputNeurons, hiddenNeurons);
        this.weightsHiddenOutput = this.initializeWeights(hiddenNeurons, outputNeurons);

        this.biasHidden = new Array(hiddenNeurons).fill(1);
        this.biasOutput = new Array(outputNeurons).fill(1);

        this.inputs = new Array(inputNeurons).fill(0);

        this.hiddenOutputs = new Array(hiddenNeurons).fill(0);
        this.hiddenOutputCounters = new Array(hiddenNeurons).fill(0).map(() => float_counter());

        this.outputOutputs = new Array(outputNeurons).fill(0);
        this.outputOutputCounters = new Array(outputNeurons).fill(0).map(() => float_counter());

        this.outputGradients = new Array(outputNeurons).fill(0);

        this.hiddenGradients = new Array(hiddenNeurons).fill(0);
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
                $.add(item_edit(this.inputNeuronCounters[j].item, 0, this.hiddenOutputCounters[i].item, TIMER, NONE, TIMER, ADD, MUL, NONE, this.weightsInputHidden[j][i]).with(obj_props.X, 75).with(obj_props.Y, 145));
                wait(0.1)
            }
            this.hiddenOutputCounters[i].add(this.biasHidden[i]);
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
                $.add(item_edit(this.hiddenOutputCounters[j].item, 0, this.outputOutputCounters[i].item, TIMER, TIMER, TIMER, ADD, MUL, NONE, this.weightsHiddenOutput[j][i]).with(obj_props.X, 145).with(obj_props.Y, 145));
                wait(0.1)
            }
            this.outputOutputCounters[i].add(this.biasOutput[i]);
            $.add(item_edit(undefined, undefined, this.outputOutputCounters[i].item, NONE, NONE, TIMER, MUL, NONE, NONE, 1, NONE, NONE, NONE, RND).with(obj_props.X, 115).with(obj_props.Y, 145)); // rounds this.outputOutputCounters[i]
        }
    }

    feedForward(inputData) {
        this.setInputData(inputData);
        this.passThroughHiddenLayer();
        this.passThroughOutputLayer();
    }

    GD_feedForward(inputData) {
        this.GD_setInputData(inputData);
        wait(0.1)
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

    passErrorToHiddenLayer() {
        for (let i = 0; i < this.hiddenNeurons; i++) {
            let errorGradient = 0;
            for (let j = 0; j < this.outputNeurons; j++) {
                errorGradient += this.outputGradients[j] * this.weightsHiddenOutput[i][j];
            }
            this.hiddenGradients[i] = errorGradient * (this.hiddenOutputs[i] > 0 ? 1 : 0);
        }
    }

    updateWeights(learningRate) {
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
}

const neuralNetwork = new NeuralNetwork(3, 9, 1);

const trainingData = [
    { input: [1, 2, 3], output: [4] },
    { input: [1, 2, 4], output: [8] },
    { input: [5, 10, 15], output: [20] },
    { input: [1, 1, 2], output: [3] },
];

const iterations = 10000;
for (let i = 0; i < iterations; i++) {
    for (const data of trainingData) {
        neuralNetwork.feedForward(data.input);
        const error = neuralNetwork.calculateError(data.output);
        neuralNetwork.passErrorToOutputLayer(data.output);
        neuralNetwork.passErrorToHiddenLayer();
        neuralNetwork.updateWeights(0.001);
    }
}

neuralNetwork.inputNeuronCounters.forEach((x, i) => x.display(i * 70 + 45, 105))
neuralNetwork.hiddenOutputCounters.forEach((x, i) => x.display(i * 70 + 45, 135))
neuralNetwork.outputOutputCounters.forEach((x, i) => x.display(i * 70 + 45, 75))

wait(1);
neuralNetwork.feedForward([1, 2, 3]);
neuralNetwork.GD_feedForward([1, 2, 3]);