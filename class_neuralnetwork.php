<?php
    class NeuralNetwork {
        protected $nodeCount = array ();
        protected $nodeValue = array ();
        protected $nodeThreshold = array ();
        protected $edgeWeight = array ();
        protected $learningRate = array (0.1);
        protected $layerCount = 0;
        protected $previousWeightCorrection = array ();
        protected $momentum = 0.8;
        protected $isVerbose = true;
        protected $weightsInitialized = false;

        public $trainInputs = array ();
        public $trainOutput = array ();
        public $trainDataID = array ();

        public $controlInputs = array ();
        public $controlOutput = array ();
        public $controlDataID = array ();

        protected $epoch;
        protected $errorTrainingset;
        protected $errorControlset;
        protected $success;

        public function __construct($nodeCount){
            if (!is_array($nodeCount)){
                $nodeCount = func_get_args();
            }
            $this->nodeCount = $nodeCount;
            $this->layerCount = count($this->nodeCount);
        }
        public function getNodeWeight($layer, $nodeSource, $nodeDestination ){
            return $this->edgeWeight[$layer][$nodeSource][$nodeDestination];
        }
        public function getNodeCount($layer){
            return $this->nodeCount[$layer];
        }
        public function getNodeValue($layer, $node){
            return $this->nodeValue[$layer][$node];
        }
        public function export(){
            return array(
                'layerCount' => $this->layerCount,
                'nodeCount' => $this->nodeCount,
                'edgeWeight' => $this->edgeWeight,
                'nodeThreshold' => $this->nodeThreshold,
                'learningRate' => $this->learningRate,
                'momentum' => $this->momentum,
                'isVerbose' => $this->isVerbose,
                'weightsInitialized' => $this->weightsInitialized,
            );
        }
        public function import($nn_array){
            foreach ($nn_array as $key => $value){
                $this->$key = $value;
            }
            return $this;
        }
        public function setLearningRate($learningRate){
            if (!is_array($learningRate)){
                $learningRate = func_get_args();
            }
            $this->learningRate = $learningRate;
        }
        public function getLearningRate($layer){
            if (array_key_exists($layer, $this->learningRate)){
                return $this->learningRate[$layer];
            }
            return $this->learningRate[0];
        }
        public function setMomentum($momentum){
            $this->momentum = $momentum;
        }
        public function getMomentum(){
            return $this->momentum;
        }
        public function calculate($input){
            foreach ($input as $index => $value){
                $this->nodeValue[0][$index] = $value;
            }
            for ($layer = 1; $layer < $this->layerCount; $layer ++){
                $prev_layer = $layer -1;
                for ($node = 0; $node < ($this->nodeCount[$layer]); $node ++){
                    $node_value = 0.0;
                    for ($prev_node = 0; $prev_node < ($this->nodeCount[$prev_layer]); $prev_node ++) {
                        $inputnode_value = $this->nodeValue[$prev_layer][$prev_node];
                        $edge_weight = $this->edgeWeight[$prev_layer][$prev_node][$node];
                        $node_value = $node_value + ($inputnode_value * $edge_weight);
                    }
                    $node_value = $node_value - $this->nodeThreshold[$layer][$node];
                    $node_value = $this->activation($node_value);
                    $this->nodeValue[$layer][$node] = $node_value;
                }
            }
            return $this->nodeValue[$this->layerCount - 1];
        }
        protected function activation($value){
            return tanh($value);
            // return (1.0 / (1.0 + exp(- $value))); Sigmoid Activation
        }
        protected function derivativeActivation($value){
            $tanh = tanh($value);
            return 1.0 - $tanh * $tanh;
            //return $value * (1.0 - $value); Derivative Sigmoid Activation
        }
        public function addTestData($input, $output, $id = null){
            $index = count($this->trainInputs);
            foreach ($input as $node => $value){
                $this->trainInputs[$index][$node] = $value;
            }
            foreach ($output as $node => $value){
                $this->trainOutput[$index][$node] = $value;
            }
            $this->trainDataID[$index] = $id;
        }
        public function getTestDataIDs(){
            return $this->trainDataID;
        }
        public function addControlData($input, $output, $id = null){
            $index = count($this->controlInputs);
            foreach ($input as $node => $value) {
                $this->controlInputs[$index][$node] = $value;
            }

            foreach ($output as $node => $value){
                $this->controlOutput[$index][$node] = $value;
            }

            $this->controlDataID[$index] = $id;
        }
        public function getControlDataIDs(){
            return $this->controlDataID;
        }
        public function showWeights($force = false){
            if ($this->isVerbose() || $force){
                echo "<hr>";
                echo "<br />Weights: <pre>".print_r($this->edgeWeight, true)."</pre>";
                echo "<br />Thresholds: <pre>".print_r($this->nodeThreshold, true)."</pre>";
            }
        }
        public function setVerbose($isVerbose){
            $this->isVerbose = $isVerbose;
        }
        public function isVerbose(){
            return $this->isVerbose;
        }
        public function load($filename){
            if (file_exists($filename)){
                $data = parse_ini_file($filename);
                if (array_key_exists("edges", $data) && array_key_exists("thresholds", $data)){
                    $this->initWeights();
                    $this->edgeWeight = unserialize($data['edges']);
                    $this->nodeThreshold = unserialize($data['thresholds']);
                    $this->weightsInitialized = true;
                    if (array_key_exists("training_data", $data) && array_key_exists("control_data", $data)){
                        $this->trainDataID = unserialize($data['training_data']);
                        $this->controlDataID = unserialize($data['control_data']);
                        $this->controlInputs = array ();
                        $this->controlOutput = array ();
                        $this->trainInputs = array ();
                        $this->trainOutput = array ();
                    }
                    return true;
                }
            }
            return false;
        }
        public function save($filename){
            $f = fopen($filename, "w");
            if ($f){
                fwrite($f, "[weights]");
                fwrite($f, "\r\nedges = \"".serialize($this->edgeWeight)."\"");
                fwrite($f, "\r\nthresholds = \"".serialize($this->nodeThreshold)."\"");
                fwrite($f, "\r\n");
                fwrite($f, "[identifiers]");
                fwrite($f, "\r\ntraining_data = \"".serialize($this->trainDataID)."\"");
                fwrite($f, "\r\ncontrol_data = \"".serialize($this->controlDataID)."\"");
                fclose($f);
                return true;
            }
            return false;
        }
        public function clear(){
            $this->initWeights();
        }
        public function train($maxEpochs = 500, $maxError = 0.01){
            if (!$this->weightsInitialized){
                $this->initWeights();
            }
            if ($this->isVerbose()){
                echo "<table>";
                echo "<tr><th>#</th><th>error(trainingdata)</th><th>error(controldata)</th><th>slope(error(controldata))</th></tr>";
            }
            $epoch = 0;
            $errorControlSet = array ();
            $avgErrorControlSet = array ();
            $sample_count = 10;
            do {
                for ($i = 0; $i < count($this->trainInputs); $i ++){
                    $index = mt_rand(0, count($this->trainInputs) - 1);
                    $input = $this->trainInputs[$index];
                    $desired_output = $this->trainOutput[$index];
                    $output = $this->calculate($input);
                    $this->backpropagate($output, $desired_output);
                }
                //set_time_limit(300);
                $squaredError = $this->squaredErrorEpoch();
                if ($epoch % 2 == 0){
                    $squaredErrorControlSet = $this->squaredErrorControlSet();
                    $errorControlSet[] = $squaredErrorControlSet;
                    if (count($errorControlSet) > $sample_count){
                        $avgErrorControlSet[] = array_sum(array_slice($errorControlSet, -$sample_count)) / $sample_count;
                    }
                    list ($slope, $offset) = $this->fitLine($avgErrorControlSet);
                    $controlset_msg = $squaredErrorControlSet;
                } else {
                    $controlset_msg = "";
                }
                if ($this->isVerbose()){
                    echo "<tr><td><b>$epoch</b></td><td>$squaredError</td><td>$controlset_msg";
                    echo "<script type='text/javascript'>window.scrollBy(0,100);</script>";
                    echo "</td><td>$slope</td></tr>";
                    echo "</td></tr>";
                    //flush();
                    //ob_flush();
                }
                $stop_1 = $squaredError <= $maxError || $squaredErrorControlSet <= $maxError;
                $stop_2 = $epoch ++ > $maxEpochs;
                $stop_3 = $slope > 0;
            } while (!$stop_1 && !$stop_2 && !$stop_3);
            $this->setEpoch($epoch);
            $this->setErrorTrainingSet($squaredError);
            $this->setErrorControlSet($squaredErrorControlSet);
            $this->setTrainingSuccessful($stop_1);
            if ($this->isVerbose()) {
                echo "</table>";
            }
            return $stop_1;
        }
        private function setEpoch($epoch){
            $this->epoch = $epoch;
        }
        public function getEpoch(){
            return $this->epoch;
        }
        private function setErrorTrainingSet($error){
            $this->errorTrainingset = $error;
        }
        public function getErrorTrainingSet(){
            return $this->errorTrainingset;
        }
        private function setErrorControlSet($error){
            $this->errorControlset = $error;
        }
        public function getErrorControlSet(){
            return $this->errorControlset;
        }
        private function setTrainingSuccessful($success){
            $this->success = $success;
        }
        public function getTrainingSuccessful(){
            return $this->success;
        }
        private function fitLine($data){
            $n = count($data);
            if ($n > 1) {
                $sum_y = 0;
                $sum_x = 0;
                $sum_x2 = 0;
                $sum_xy = 0;
                foreach ($data as $x => $y){
                    $sum_x += $x;
                    $sum_y += $y;
                    $sum_x2 += $x * $x;
                    $sum_xy += $x * $y;
                }
                $offset = ($sum_y * $sum_x2 - $sum_x * $sum_xy) / ($n * $sum_x2 - $sum_x * $sum_x);
                $slope = ($n * $sum_xy - $sum_x * $sum_y) / ($n * $sum_x2 - $sum_x * $sum_x);
                return array ($slope, $offset);
            } else {
                return array (0.0, 0.0);
            }
        }
        private function getRandomWeight($layer){
            return ((mt_rand(0, 1000) / 1000) - 0.5) / 2;
        }
        private function initWeights(){
            for ($layer = 1; $layer < $this->layerCount; $layer ++){
                $prev_layer = $layer -1;
                for ($node = 0; $node < $this->nodeCount[$layer]; $node ++){
                    $this->nodeThreshold[$layer][$node] = $this->getRandomWeight($layer);
                    for ($prev_index = 0; $prev_index < $this->nodeCount[$prev_layer]; $prev_index ++){
                        $this->edgeWeight[$prev_layer][$prev_index][$node] = $this->getRandomWeight($prev_layer);
                        $this->previousWeightCorrection[$prev_layer][$prev_index] = 0.0;
                    }
                }
            }
        }
        private function backpropagate($output, $desired_output){
            $errorgradient = array ();
            $outputlayer = $this->layerCount - 1;
            $momentum = $this->getMomentum();
            for ($layer = $this->layerCount - 1; $layer > 0; $layer --){
                for ($node = 0; $node < $this->nodeCount[$layer]; $node ++){
                    if ($layer == $outputlayer) {
                        $error = $desired_output[$node] - $output[$node];
                        $errorgradient[$layer][$node] = $this->derivativeActivation($output[$node]) * $error;
                    } 
                    else{
                        $next_layer = $layer +1;
                        $productsum = 0;
                        for ($next_index = 0; $next_index < ($this->nodeCount[$next_layer]); $next_index ++){
                            $_errorgradient = $errorgradient[$next_layer][$next_index];
                            $_edgeWeight = $this->edgeWeight[$layer][$node][$next_index];
                            $productsum = $productsum + $_errorgradient * $_edgeWeight;
                        }
                        $nodeValue = $this->nodeValue[$layer][$node];
                        $errorgradient[$layer][$node] = $this->derivativeActivation($nodeValue) * $productsum;
                    }
                    $prev_layer = $layer -1;
                    $learning_rate = $this->getlearningRate($prev_layer);
                    for ($prev_index = 0; $prev_index < ($this->nodeCount[$prev_layer]); $prev_index ++){
                        $nodeValue = $this->nodeValue[$prev_layer][$prev_index];
                        $edgeWeight = $this->edgeWeight[$prev_layer][$prev_index][$node];
                        $weight_correction = $learning_rate * $nodeValue * $errorgradient[$layer][$node];
                        $prev_weightcorrection = @$this->previousWeightCorrection[$layer][$node];
                        $new_weight = $edgeWeight + $weight_correction + $momentum * $prev_weightcorrection;
                        $this->edgeWeight[$prev_layer][$prev_index][$node] = $new_weight;
                        $this->previousWeightCorrection[$layer][$node] = $weight_correction;
                    }
                    $threshold_correction = $learning_rate * -1 * $errorgradient[$layer][$node];
                    $new_threshold = $this->nodeThreshold[$layer][$node] + $threshold_correction;
                    $this->nodeThreshold[$layer][$node] = $new_threshold;
                }
            }
        }
        private function squaredErrorEpoch(){
            $RMSerror = 0.0;
            for ($i = 0; $i < count($this->trainInputs); $i ++){
                $RMSerror += $this->squaredError($this->trainInputs[$i], $this->trainOutput[$i]);
            }
            $RMSerror = $RMSerror / count($this->trainInputs);
            return sqrt($RMSerror);
        }
        private function squaredErrorControlSet(){
            if (count($this->controlInputs) == 0){
                return 1.0;
            }
            $RMSerror = 0.0;
            for ($i = 0; $i < count($this->controlInputs); $i ++){
                $RMSerror += $this->squaredError($this->controlInputs[$i], $this->controlOutput[$i]);
            }
            $RMSerror = $RMSerror / count($this->controlInputs);
            return sqrt($RMSerror);
        }
        private function squaredError($input, $desired_output){
            $output = $this->calculate($input);
            $RMSerror = 0.0;
            foreach ($output as $node => $value){
                $error = $output[$node] - $desired_output[$node];
                $RMSerror = $RMSerror + ($error * $error);
            }
            return $RMSerror;
        }
    }
?>
