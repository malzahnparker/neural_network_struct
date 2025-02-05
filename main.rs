use rand::distr::{Distribution, Uniform};
use rand::thread_rng;

struct NeuralNetworkLayer {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    num_inputs: usize,
    num_outputs: usize,
    outputs: Vec<f32>,
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
            outputs: vec![0.0; num_outputs],
        }

    }
    fn forward(&self, inputs: &[f32]) -> Vec<f32> {    
        for i in 0..self.num_outputs {
            self.outputs[i] = 0.0;
            for j in 0..self.num_inputs {
                self.outputs[i] += inputs[j] * self.weights[j][i];
            }
            self.outputs[i] += self.biases[i];
            self.outputs[i] = sigmoid(self.outputs[i]);
        }
        return self.outputs;
    }
    /*fn backwards(previous_errors: Vec<f32>) {
        let mut hidden_errors = vec![0.0; HIDDEN_SIZE];
        for j in 0..HIDDEN_SIZE {
            for k in 0..OUTPUT_SIZE {
                hidden_errors[j] += output_errors[k] * self.weights_hidden_output[j][k];
            }
            hidden_errors[j] *= sigmoid_derivative(self.hidden_layer[j]);
        }
    }*/
}


fn main() {
    let dense1 = NeuralNetworkLayer::new(8, 16);
    let dense2 = NeuralNetworkLayer::new(16, 16);
    let dense3 = NeuralNetworkLayer::new(16, 2);
    let inputs = vec![0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
    dense1.forward(&inputs);
    dense2.forward(&dense1.outputs);
    dense3.forward(&dense2.outputs);
    dbg!("{}", dense3.outputs);
}
