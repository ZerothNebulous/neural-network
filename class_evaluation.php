<?php
  class Evaluation{
    private $outputData = array ();
    private $outputTarget = array ();
    
    private $truePositive = 0;
    private $falsePositive = 0;
    private $trueNegative = 0;
    private $falseNegative = 0;
    
    public function __constructor($outputData, $outputTarget){
      $this->outputData = $outputData;
      $this->outputTarget = $outputTarget;
    }
    
    public function getPrecision(){
    
    }
    
    public function getSensitivity(){
    
    }
    
  }

?>
