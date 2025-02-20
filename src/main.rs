use rand::distr::{Distribution, Uniform};
use rand::thread_rng;

const LEARNING_RATE: f32 = 0.01;

struct NeuralNetworkLayer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    num_inputs: usize,
    num_outputs: usize,
    inputs: Vec<f32>,
    outputs: Vec<f32>,
    errors: Vec<f32>,
}

impl NeuralNetworkLayer {
    fn new(num_inputs: usize, num_outputs: usize) -> Self {
        let mut rng = thread_rng();
        let between = Uniform::try_from(-1.0..1.0).unwrap();

        let weights = (0..num_inputs)
            .map(|_| (0..num_outputs).map(|_| between.sample(&mut rng)).collect())
            .collect();

        let biases = (0..num_outputs).map(|_| between.sample(&mut rng)).collect();

        NeuralNetworkLayer {
            weights: weights,
            biases: biases,
            num_inputs: num_inputs,
            num_outputs: num_outputs,
            inputs: vec![0.0; num_inputs],
            outputs: vec![0.0; num_outputs],
            errors: vec![0.0; num_outputs]
        }

    }
    fn forward(&mut self, inputs: &[f32]) {    
        self.inputs = inputs.clone().to_vec();
        for i in 0..self.num_outputs {
            self.outputs[i] = 0.0;
            for j in 0..self.num_inputs {
                self.outputs[i] += inputs[j] * self.weights[j][i];
            }
            self.outputs[i] += self.biases[i];
            self.outputs[i] = self.sigmoid(self.outputs[i]);
        }
    }
    fn backwards(&mut self, previous_errors: &Vec<f32>, previous_weights: &Vec<Vec<f32>>, first: bool) {
        if first {
            self.errors = previous_errors.clone();
        } else {
        self.errors = vec![0.0; self.num_outputs];
        for j in 0..self.num_outputs {
            for k in 0..previous_errors.len() {
                self.errors[j] += previous_errors[k] * previous_weights[j][k];
            }
            self.errors[j] *= self.sigmoid_derivative(self.outputs[j]);
        }
    }
        for j in 0..self.num_outputs {
            for i in 0..self.num_inputs {
                self.weights[i][j] += LEARNING_RATE * self.errors[j] * self.inputs[i];
            }
            self.biases[j] += LEARNING_RATE * self.errors[j];
        }
    }
    fn sigmoid(&self, x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    
    fn sigmoid_derivative(&self, x: f32) -> f32 {
        self.sigmoid(x) * (self.sigmoid(x) - 1.0)
    }
}


fn main() {
    let mut dense1 = NeuralNetworkLayer::new(8, 16);
    let mut dense2 = NeuralNetworkLayer::new(16, 16);
    let mut dense3 = NeuralNetworkLayer::new(16, 2);
    let inputs = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    let errors = vec![1.0 - dense3.outputs[0], 0.5 - dense3.outputs[1]];
    let null = vec![vec![0.0]];
    for i in 0..100000 {
    dense1.forward(&inputs);
    dense2.forward(&dense1.outputs);
    dense3.forward(&dense2.outputs);
    println!{"x: {}, y: {}", dense3.outputs[0], dense3.outputs[1]};
    dense3.backwards(&errors, &null, true);
    dense2.backwards(&dense3.errors, &dense3.weights, false);
    dense1.backwards(&dense2.errors, &dense2.weights, false);
    }

}
