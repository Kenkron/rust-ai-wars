use vai::*;
use crate::nn::CellNN;

#[derive(Clone)]
pub struct VaiNet<
    const I: usize,
    const O: usize,
    const HIDDEN: usize>
{
    layers: VAI<I,O,HIDDEN,0>
}

impl<
    const I: usize,
    const O: usize,
    const HIDDEN: usize>
VaiNet<I,O,HIDDEN> {
    pub fn new() -> Self {
        Self {
            layers: VAI::<I, O, HIDDEN, 0>::new().create_variant(5.)
        }
    }
}

impl<
    const I: usize,
    const O: usize,
    const HIDDEN: usize>
CellNN for VaiNet<I,O,HIDDEN> {
    fn predict(&self, inputs: &Vec<f64>)
    -> Vec<Vec<f64>> {
        // translate input to f32
        let mut input_f32: Vec<f32> = inputs.iter().map(|x| x.to_owned() as f32).collect();
        // Add a constant
        input_f32.push(1.0);
        // translate output to f64
        self.layers.process_slice_transparent(&input_f32).iter()
            .map(|layer| layer.iter().map(|x| x.to_owned() as f64).collect())
            .collect()
    }
    fn mutate(&mut self) {
        self.layers.create_variant(1.0);
    }
}