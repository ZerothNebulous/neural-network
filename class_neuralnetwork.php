<?php
    class NeuralNetwork{
        private $layerCount = 0;
        private $neuronCount = array (); //neuronCount[layer]
        private $neuronValue = array (); //neuronValue[layer][index]
        private $neuronWeight = array (); //neuronWeight[layer][source][destination]
        private $neuronThreshold = array (); //neuronThreshold[layer][index]
        private $neuronWeightCorrection = array (); //neuronWeightCorrection[layer][index]

        //Parameters
        private $epoch = 500;
        private $learningRate = array (0.1); //learningRate[layer]
        private $momentum = 0.8;
        private $weightsInitialized = false;
        public $success;
        public $useBiasNeuron = false;
        public $useSigmoidActivation = false;

        //Data
        public $trainingInput = array (); //trainingInput[layer][index]
        public $trainingOutput = array (); //trainingInput[layer][index]
        public $trainingSetRMSE = 0.0;

        //NeuralNetwork(InputCount, HiddenCount, ..., OutputCount)
        public function __construct($value){
            if (!is_array($value)){
                $value = func_get_args();
            }
            $this->neuronCount = $value;
            $this->layerCount = count($this->neuronCount);
            for($layer = 0; $layer < $this->layerCount - 1; $layer++){
                $this->setBias($layer, 1.0); //Adds Bias Neuron per input/hidden layer.
            }
        }
        //Parameter Functions
        public function setEpoch($value){
            $this->epoch = $value;
        }

        public function getEpoch(){
            return $this->epoch;
        }

        public function setLearningRate($value){
            if(!is_array($value)){
                $value = func_get_args();
            }
            $this->learningRate = $value;
        }

        public function getLearningRate($layer){
            if(array_key_exists($layer, $this->learningRate)){
                return $this->learningRate[$layer];
            }
            return $this->learningRate[0];
        }

        public function setMomentum($value){
            $this->momentum = $value;
        }

        public function getMomentum(){
            return $this->momentum;
        }

        public function setBias($layer, $value){
            $neuron = $this->neuronCount[$layer];
            $this->neuronValue[$layer][$neuron] = $value;
        }

        protected function getRandomWeight($seed){
            return ((mt_rand(0, 1000) / 1000) - 0.5) / 2;
        }

        private function setErrorTrainingSet($value){
            $this->errorTrainingSet = $value;
        }

        public function getErrorTrainingSet(){
            return $this->errorTrainingSet;
        }

        private function setTrainingSuccessful($condition){
            $this->success = $condition;
        }

        public function isTrainingSuccessful(){
            return $this->success;
        }

        //Initialize Weights including Bias and Threshold
        private function initializeWeights(){
            for($layer = 1; $layer < $this->layerCount; $layer++){
                $previousLayer = $layer - 1;
                for($neuron = 0; $neuron < $this->neuronCount[$layer]; $neuron++){
                    $this->neuronThreshold[$layer][$neuron] = $this->getRandomWeight($layer);
                    for($previousNeuron = 0; $previousNeuron <= $this->neuronCount[$previousLayer]+1; $previousNeuron++){
                        $this->neuronWeight[$previousLayer][$previousNeuron][$neuron] = $this->getRandomWeight($layer);
                        $this->neuronWeightCorrection[$previousLayer][$previousNeuron] = 0.0;
                    }
                }
            }
        }

        public function addTrainingData($inputData, $outputData){
            $index = count($this->trainingInput);
            foreach($inputData as $attribute => $value){
                $this->trainingInput[$index][$attribute] = $value;
            }
            foreach ($outputData as $attribute => $value){
                $this->trainingOutput[$index][$attribute] = $value;
            }
        }

        //Calculation Functions
        public function feedforward($inputData){
            foreach($inputData as $index => $value){
                $this->neuronValue[0][$index] = $value;
            }
            for($layer = 1; $layer < $this->layerCount; $layer++){
                $previousLayer = $layer - 1;
                for($neuron = 0; $neuron < $this->neuronCount[$layer]; $neuron++){
                    $weightedSum = 0.0;
                    $previousNeuronCount = $this->neuronCount[$previousLayer];
                    if($this->useBiasNeuron){
                        $previousNeuronCount += 1;
                    }
                    for($previousNeuron = 0; $previousNeuron < $previousNeuronCount; $previousNeuron++){
                        $value = $this->neuronValue[$previousLayer][$previousNeuron];
                        $weight = $this->neuronWeight[$previousLayer][$previousNeuron][$neuron];
                        $weightedSum += $value * $weight;
                    }
                    if(!$this->useBiasNeuron){
                        $weightedSum -= $this->neuronThreshold[$layer][$neuron];
                    }
                    $this->neuronValue[$layer][$neuron] = $this->activation($weightedSum);
                }
            }
            return $this->neuronValue[$this->layerCount - 1];
        }

        private function backpropagate($outputNet, $outputTarget){
            $errorGradient = array ();
            $outputLayer = $this->layerCount - 1;
            $momentum = $this->getMomentum();
            for($layer = $outputLayer; $layer > 0; $layer--){
                $neuronCount = $this->neuronCount[$layer];
                for($neuron = 0; $neuron < $neuronCount; $neuron++){
                    if($layer == $outputLayer){
                        $error = $outputTarget[$neuron] - $outputNet[$neuron];
                        $errorGradient[$layer][$neuron] = $this->derivativeActivation($outputNet[$neuron]) * $error;
                    }
                    else{
                        $nextLayer = $layer + 1;
                        $productSum = 0.0;
                        $nextNeuronCount = $this->neuronCount[$nextLayer];
                        $previousNeuron = $neuron;
                        if($this->useBiasNeuron){
                            $previousNeuron += 1;
                        }
                        for($nextNeuron = 0; $nextNeuron < $nextNeuronCount; $nextNeuron++){
                            $neuronErrorGradient = $errorGradient[$nextLayer][$nextNeuron];
                            $neuronWeight = $this->neuronWeight[$layer][$previousNeuron][$nextNeuron];
                            $productSum += $neuronErrorGradient * $neuronWeight;
                        }
                        $neuronValue = $this->neuronValue[$layer][$previousNeuron];
                        $errorGradient[$layer][$neuron] = $this->derivativeActivation($neuronValue) * $productSum;
                    }
                    $previousLayer = $layer - 1;
                    $learningRate = $this->getLearningRate($previousLayer);
                    $previousNeuronCount = $this->neuronCount[$previousLayer];
                    if($this->useBiasNeuron){
                        $previousNeuronCount += 1;
                    }
                    for($previousNeuron = 0; $previousNeuron < $previousNeuronCount; $previousNeuron++){
                        $neuronValue = $this->neuronValue[$previousLayer][$previousNeuron];
                        $neuronWeight = $this->neuronWeight[$previousLayer][$previousNeuron][$neuron];
                        $weightCorrection = $learningRate * $neuronValue * $errorGradient[$layer][$neuron];
                        $previousWeightCorrection = @$this->neuronWeightCorrection[$layer][$neuron];
                        $newWeight = $neuronWeight + $weightCorrection + $momentum * $previousWeightCorrection;
                        $this->neuronWeight[$previousLayer][$previousNeuron][$neuron] = $newWeight;
                        $this->previousWeightCorrection[$layer][$neuron] = $weightCorrection;
                    }
                    if(!$this->useBiasNeuron){
                        $thresholdCorrection = $learningRate * -1 * $errorGradient[$layer][$neuron];
                        $this->neuronThreshold[$layer][$neuron] += $thresholdCorrection;
                    }
                }
            }
        }

        public function train($maxEpoch = 500, $maxRMSE = 0.01){
            if(!$this->weightsInitialized){
                $this->initializeWeights();
            }
            $rmse = 0.0;
            $epoch = 0;
            do{
                for($i = 0; $i < count($this->trainingInput); $i++){
                    $index = mt_rand(0, count($this->trainingInput) - 1);
                    $inputData = $this->trainingInput[$index];
                    $outputTarget = $this->trainingOutput[$index];
                    $outputNet = $this->feedforward($inputData);
                    $this->backpropagate($outputNet, $outputTarget);
                }
                $rmse = $this->trainingSetRMSE();
                $condition1 = $rmse <= $maxRMSE;
                $condition2 = $epoch++ > $maxEpoch;
            } while(!$condition1 && !$condition2);
            $this->setEpoch($epoch);
            $this->setErrorTrainingSet($rmse);
            $this->setTrainingSuccessful(!$condition1);
            return $condition1;
        }

        //Root Mean Squared Error on Training set
        private function trainingSetRMSE(){
            $rmse = 0.0;
            for($index = 0; $index < count($this->trainingInput); $index++){
                $rmse += $this->feedforwardRMSE($this->trainingInput[$index], $this->trainingOutput[$index]);
            }
            $rmse /= count($this->trainingInput);
            return sqrt($rmse);
        }
        
        //Root Mean Squared Error per Feedforward
        private function feedforwardRMSE($inputData, $outputTarget){
            $outputNet = $this->feedforward($inputData);
            $rmse = 0.0;
            foreach($outputNet as $neuron => $value){
                $error = $outputNet[$neuron] - $outputTarget[$neuron];
                $rmse += ($error * $error);
            }
            return $rmse;
        }

        private function activation($value){
            if($this->useSigmoidActivation){
                return 1.0 / (1.0 + exp(- $value));
            }
            else{
                return tanh($value);
            }
        }

        private function derivativeActivation($value){
            if($this->useSigmoidActivation){
                return $value * (1.0 - $value);
            }
            else{
                $tanh = tanh($value);
                return 1.0 - $tanh * $tanh;
            }
        }

    }
?>
