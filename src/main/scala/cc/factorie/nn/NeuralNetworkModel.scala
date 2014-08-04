package cc.factorie.nn

import cc.factorie.la.{WeightsMapAccumulator, Tensor1}
import cc.factorie.model
import cc.factorie.model._
import cc.factorie.optimize.{Example, MultivariateOptimizableObjective}
import cc.factorie.optimize.OptimizableObjectives.SquaredMultivariate
import cc.factorie.util.{FastLogging, DoubleAccumulator}
import cc.factorie.variable.Var

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random


trait NeuralNetworkModel extends Model with Parameters {
  implicit val rng = new Random
  type Factor = NeuralNetworkWeightsFamily#Factor //easier access
}


//Usage: first createInputOutputLayerPairs, and train using any trainer and optimizer with BackpropagationExample, which takes as input an Iterable of InputLayers.
// this mixin can handle any kind of network structure that is a DAG
trait FeedForwardNeuralNetworkModel extends NeuralNetworkModel with FastLogging {
  type InputLayer <: InputNeuralNetworkLayer
  type OutputLayer <: LabeledNeuralNetworkLayer
  type Input
  type Output

  def totalObjective(inputLayers:Iterable[InputLayer],outputLayers:Iterable[OutputLayer]) = {
    forwardPropagateInput(inputLayers)
    outputLayers.map(o => {
      o.error()
      o.lastObjective
    }).sum
  }

  def forwardPropagateInput(inputLayers:Iterable[InputLayer]) = {
    val orderedFactors = calculateComputationSeq(inputLayers)
    _forwardPropagateInput(orderedFactors,inputLayers)
  }

  protected def _forwardPropagateInput(orderedFactors:Seq[Seq[Factor]], inputLayers:Iterable[InputLayer]) = {
    val updated = mutable.Set[NeuralNetworkLayer]() ++ inputLayers
    orderedFactors.foldLeft(mutable.Set[NeuralNetworkLayer]()){ case (zeroed,fs) =>
      fs.foreach { case f =>
        f.outputLayers.withFilter(l => !zeroed.contains(l)).foreach(l => {
          l.zeroInput()
          zeroed += l
        })
      }
      zeroed
    }
    orderedFactors.foreach(fs => {
      fs.foreach { f =>
        f.inputLayers.withFilter(l => !updated.contains(l)).foreach(l => {
          l.updateActivation()
          updated += l
        })
        f.forwardPropagate()
        fs.foreach(_.foreach(_.variables.foreach { case v =>
          if (v.isInstanceOf[OutputLayer])
            v.asInstanceOf[OutputLayer].updateActivation()
        }))
      }
    })
  }

  //returns gradient on the error function for all weights of this model
  def backPropagateOutputError(inputLayers:Iterable[InputLayer]):WeightsMap= {
    val orderedFactors = calculateComputationSeq(inputLayers)
    _backPropagateOutputError(orderedFactors)
  }

  protected def _backPropagateOutputError(orderedFactors: Seq[Seq[Factor]]): WeightsMap = {
    val map = new WeightsMap(key => key.value.blankCopy)
    orderedFactors.foldLeft(mutable.Set[NeuralNetworkLayer]()){ case (zeroed,fs) =>
      fs.foreach { case f =>
        f.inputLayers.withFilter(l => !zeroed.contains(l)).foreach(l => {
          l.zeroError()
          zeroed += l
        })
      }
      zeroed
    }

    val errors = orderedFactors.reverseIterator.flatMap(fs => {
      fs.map(f => f.family.weights -> f.backPropagateError)
    })
    errors.foreach{ case (weights,gradient) => map(weights) += gradient }
    map
  }

  def forwardAndBackPropagateOutputError(inputLayers:Iterable[InputLayer]):WeightsMap = {
    val orderedFactors = calculateComputationSeq(inputLayers)
    _forwardPropagateInput(orderedFactors,inputLayers)
    _backPropagateOutputError(orderedFactors)
  }

  //computes the DAG starting from the input layers and returns a seq of independent factors
  //Override this if there is a more efficient way of computing this (e.g.: see BasicFeedForwardNeuralNetwork)
  def calculateComputationSeq(inputLayers:Iterable[InputLayer]): Seq[Seq[Factor]] = {
    var currentLayers = mutable.HashSet[NeuralNetworkLayer]() ++ inputLayers
    val updatedLayers = mutable.HashSet[NeuralNetworkLayer]() ++ inputLayers
    val factorSeq = ArrayBuffer[ArrayBuffer[Factor]]()

    val updatedInputLayers = mutable.Map[Factor,(Int,Int)]()
    val updatedInputFactors = mutable.Map[NeuralNetworkLayer,(Int,Int)]()

    while(currentLayers.nonEmpty) {
      val nextFactors = ArrayBuffer[Factor]()
      val nextLayers = mutable.HashSet[NeuralNetworkLayer]()
      currentLayers.foreach(l => outputFactors(l).foreach(f => {
        val outLayers = f.outputLayers
        val (nrInputs,totalNrInputs) = updatedInputLayers.getOrElseUpdate(f,(0,f.inputLayers.size))
        if (nrInputs == totalNrInputs - 1) {
          nextFactors += f
          //update number of incoming activated factors for each output layer of this factor; if full, add it to nextFactors
          outLayers.foreach(l => {
            val (ct,totalCt) =
              updatedInputFactors.getOrElseUpdate(l, {val inFactors = inputFactors(l); (inFactors.count(_.numVariables == 1),inFactors.size) }) //initialize with number of biases
            if(ct == totalCt - 1) //this layer got full input, add it as next layer
              nextLayers += l
            else //update incoming count for this layer
              updatedInputFactors += l -> (ct+1,totalCt)
          } )
        } else {
          updatedInputLayers(f) = (nrInputs+1,totalNrInputs)
        }
      }))
      currentLayers = nextLayers
      updatedLayers ++= currentLayers

      //Add biases
      currentLayers.foreach(l => nextFactors ++= inputFactors(l).filter(_.numVariables == 1))
      if(nextFactors.nonEmpty)
        factorSeq += nextFactors
    }
    factorSeq
  }

  def inputFactors(variable: NeuralNetworkLayer):Iterable[Factor]
  def outputFactors(variable: NeuralNetworkLayer):Iterable[Factor]

  override def factors(variables: Iterable[Var]): Iterable[model.Factor] =
    variables.foldLeft(mutable.Set[model.Factor]()){ case (set,v:NeuralNetworkLayer) => (set ++ inputFactors(v)) ++ outputFactors(v) }

  //creates network input and corresponding output. It is possible that the network architecture has many input and output layers, depending on its architecture
  def createNetwork(input:Input,output:Output):(Iterable[InputLayer],Iterable[OutputLayer])

  class BackPropagationExample(inputLayers:Iterable[InputLayer], outputLayers:Iterable[OutputLayer], var withGradientCheck:Boolean = false) extends Example {
    val computationSeq = calculateComputationSeq(inputLayers) //cache that, so it doesnt have to be computed all the time
    override def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
      _forwardPropagateInput(computationSeq,inputLayers)
      val gradients = _backPropagateOutputError(computationSeq)
      if(gradient != null)
        gradients.keys.foreach(k => gradient.accumulate(k,gradients(k)))
      if(withGradientCheck) {
        withGradientCheck = false
        Example.testGradient(parameters, this, verbose = true)
        withGradientCheck = true
      }
      if(value != null)
        value.accumulate(outputLayers.foldLeft(0.0)(_ + _.lastObjective))
    }
    /*val epsilon = 0.001
    def checkGradient(gradient:WeightsMap, input:Iterable[InputLayer],output:Iterable[OutputLayer]) {
      parameters.keys.foreach(w => {
        w.value.foreachActiveElement((i,v) => {
          val g_i = gradient.apply(w)(i)
          w.value.update(i,v+epsilon)
          val e1 = totalObjective(input,output)
          w.value.update(i,v-epsilon)
          val e2 = totalObjective(input,output)
          val diff = math.abs((e1-e2)/(2*epsilon)-g_i)
          w.value.update(i,v)
          if(diff > epsilon)
            logger.warn(s"gradient check failed with difference: $diff > $epsilon")
        })
      })
    }*/
  }
}

//Example feed-forward model
class BasicFeedForwardNeuralNetwork(structure:Array[(Int,ActivationFunction)],errorFunction:MultivariateOptimizableObjective[Tensor1] = new SquaredMultivariate) extends FeedForwardNeuralNetworkModel {
  trait InputLayer extends InnerLayer with InputNeuralNetworkLayer
  type Input = Tensor1
  type Output = Tensor1

  val weights:Array[BasicLayerToLayerWeightsFamily[Layer,Layer]] = (0 until structure.length-1).map { case i =>
    val in = structure(i)._1
    val out = structure(i+1)._1
    new BasicLayerToLayerWeightsFamily[Layer,Layer] {
      override val weights: Weights2 = Weights(NNUtils.fillDense(in,out)((_,_) => Random.nextGaussian()/10))
      override type FamilyType = BasicLayerToLayerWeightsFamily[Layer,Layer]
     }
  }.toArray

  val biases:Array[Bias[Layer]] = (1 until structure.length).map { case i =>
    val out = structure(i)._1
    new Bias[Layer] {
      override val weights: Weights1 = Weights(NNUtils.fillDense(out)(_ => Random.nextGaussian()/10))
      override type FamilyType = Bias[Layer]
    }
  }.toArray

  trait Layer extends NeuralNetworkLayer {
    def index:Int
    var nextFactor:BasicLayerToLayerWeightsFamily[Layer,Layer]#FactorType = null.asInstanceOf[BasicLayerToLayerWeightsFamily[Layer,Layer]#FactorType]
    var prevFactor:BasicLayerToLayerWeightsFamily[Layer,Layer]#FactorType = null.asInstanceOf[BasicLayerToLayerWeightsFamily[Layer,Layer]#FactorType]
    lazy val bias:Bias[Layer]#FactorType = {
      if (index > 0) {
        val bias = biases(index-1)
        new bias.Factor(this)
      }
      else null
    }
  }
  class InnerLayer(override val index:Int) extends BasicNeuralNetworkLayer(structure(index)._1,structure(index)._2) with Layer {
    nextFactor = {
      if (index < structure.length - 2){
        val w = weights(index)
        new w.Factor(this, new InnerLayer(index + 1))
      }
      else if (index == structure.length - 2) {
        val w = weights(index)
        new w.Factor(this, new OutputLayer())
      }
      else null
    }
    if(nextFactor != null)
      nextFactor._2.prevFactor = nextFactor
  }

  class OutputLayer extends BasicOutputNeuralNetworkLayer(NNUtils.newDense(structure.last._1),structure(structure.length-1)._2,errorFunction) with Layer {
    val index = structure.length-1
  }

  def createNetwork(input:Tensor1, output:Tensor1):(Iterable[InputLayer],Iterable[OutputLayer]) = {
    val inputLayer = new InnerLayer(0) with InputLayer
    inputLayer.set(input)(null)
    var outputLayer:Layer = inputLayer
    while(outputLayer.nextFactor != null)
      outputLayer = outputLayer.nextFactor._2
    outputLayer.asInstanceOf[OutputLayer].target.set(output)(null)
    (Iterable(inputLayer),Iterable(outputLayer.asInstanceOf[OutputLayer]))
  }
  override def inputFactors(variable: NeuralNetworkLayer): Iterable[Factor] = variable match {
    case v:Layer =>
      var fs = List[Factor]()
      if(v.prevFactor != null)
        fs ::= v.prevFactor.asInstanceOf[Factor]
      if(v.bias!=null)
        fs ::= v.bias.asInstanceOf[Factor]
      fs
  }
  override def outputFactors(variable: NeuralNetworkLayer): Iterable[Factor] = variable match {
    case v:Layer =>
      if(v.nextFactor != null)
        List(v.nextFactor)
      else
        List[Factor]()
  }

  //The rest is only there to make some implementations more efficient in this special case of a stacked FeedForwardNetwork
  override def calculateComputationSeq(inputLayers: Iterable[InputLayer]): Seq[Seq[Factor]] = {
    var currentLayers = Seq[Layer]() ++ inputLayers
    val factorSeq = ArrayBuffer[ArrayBuffer[Factor]]()
    while(currentLayers.nonEmpty) {
      val nextFactors = currentLayers.withFilter(_.nextFactor!=null).map(_.nextFactor)
      currentLayers = nextFactors.map(_._2)
      if(nextFactors.nonEmpty)
        factorSeq += new ArrayBuffer[Factor]() ++ nextFactors.asInstanceOf[Seq[Factor]] ++ currentLayers.map(_.bias).asInstanceOf[Seq[Factor]]
    }
    factorSeq
  }
}
