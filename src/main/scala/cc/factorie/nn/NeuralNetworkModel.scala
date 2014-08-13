package cc.factorie.nn

import cc.factorie.la.{SmartGradientAccumulator, WeightsMapAccumulator, Tensor1}
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
  type OutputLayer <: OutputNeuralNetworkLayer
  type Input
  type Output

  //represents ordered execution/update sequence of factors/layers, from input to output
  type OrderedFactors = Seq[(Iterable[Factor],Iterable[NeuralNetworkLayer])]

  def totalObjective(outputLayers:Iterable[OutputLayer]) = {
    outputLayers.map(o => {
      o.objectiveGradient()
      o.lastObjective
    }).sum
  }

  def forwardPropagateInput(inputLayers:Iterable[InputLayer]) = {
    val orderedFactors = calculateComputationSeq(inputLayers)
    _forwardPropagateInput(orderedFactors,inputLayers)
  }

  protected def _forwardPropagateInput(orderedFactors:OrderedFactors, inputLayers:Iterable[InputLayer]) = {
    orderedFactors.foreach(_._2.foreach(_.zeroInput()))
    orderedFactors.foreach(fs => {
      fs._1.foreach { _.forwardPropagate() }
      fs._2.foreach(_.updateActivation())
    })
  }

  //returns gradient on the error function for all weights of this model
  def backPropagateOutputGradient(inputLayers:Iterable[InputLayer]):WeightsMap= {
    val orderedFactors = calculateComputationSeq(inputLayers)
    _backPropagateOutputGradient(orderedFactors)
  }

  protected def _backPropagateOutputGradient(orderedFactors: OrderedFactors): WeightsMap = {
    val map = new WeightsMap(key => key.value.blankCopy)
    orderedFactors.foreach(_._2.withFilter(!_.isInstanceOf[OutputLayer]).foreach(_.zeroObjectiveGradient()))
    val gradients = orderedFactors.reverseIterator.flatMap(fs => {
      //multiply accumulated gradient with derivative of activation
      fs._2.foreach(_.updateObjectiveGradient())
      //backpropagate gradient
      val e = fs._1.map(f => f.family.weights -> f.backPropagateGradient)
      e
    })
    gradients.foreach{
      case (weights,gradient) =>
        map(weights) += gradient
    }
    map
  }

  protected def _backPropagateOutputGradient(orderedFactors: OrderedFactors,accumulator:WeightsMapAccumulator,scale:Double=1.0) = {
    orderedFactors.foreach(_._2.withFilter(!_.isInstanceOf[OutputLayer]).foreach(_.zeroObjectiveGradient()))
    val gradients = orderedFactors.reverseIterator.flatMap(fs => {
      //multiply accumulated gradient with derivative of activation
      fs._2.foreach(_.updateObjectiveGradient())
      //backpropagate gradient
      val e = fs._1.map(f => f.family.weights -> f.backPropagateGradient)
      e
    })
    gradients.foreach{
      case (weights,gradient) =>
        if(scale ==1.0) accumulator.accumulate(weights, gradient,scale)
        else accumulator.accumulate(weights, gradient)
    }
  }

  def forwardAndBackPropagateOutputGradient(inputLayers:Iterable[InputLayer]):WeightsMap = {
    val orderedFactors = calculateComputationSeq(inputLayers)
    _forwardPropagateInput(orderedFactors,inputLayers)
    _backPropagateOutputGradient(orderedFactors)
  }

  //computes the DAG starting from the input layers and returns a seq of independent factors and layers which inputs comes only from factors up to this point
  //Override this if there is a more efficient way of computing this (e.g.: see BasicFeedForwardNeuralNetwork)
  def calculateComputationSeq(inputLayers:Iterable[InputLayer]): OrderedFactors = {
    var currentLayers = mutable.HashSet[NeuralNetworkLayer]() ++ inputLayers
    val updatedLayers = mutable.HashSet[NeuralNetworkLayer]() ++ inputLayers
    val factorSeq = ArrayBuffer[(ArrayBuffer[Factor],Iterable[NeuralNetworkLayer])]()

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
        } else
          updatedInputLayers(f) = (nrInputs+1,totalNrInputs)
      }))
      currentLayers = nextLayers
      updatedLayers ++= currentLayers

      //Add biases
      currentLayers.foreach(l => nextFactors ++= inputFactors(l).filter(_.numVariables == 1))
      if(nextFactors.nonEmpty)
        factorSeq += ((nextFactors,nextLayers))
    }
    factorSeq
  }

  def inputFactors(variable: NeuralNetworkLayer):Iterable[Factor]
  def outputFactors(variable: NeuralNetworkLayer):Iterable[Factor]

  override def factors(variables: Iterable[Var]): Iterable[model.Factor] =
    variables.foldLeft(mutable.Set[model.Factor]()){ case (set,v:NeuralNetworkLayer) => (set ++ inputFactors(v)) ++ outputFactors(v) }

  //creates network input and corresponding output. It is possible that the network architecture has many input and output layers, depending on its architecture
  def createNetwork(input:Input,output:Output):(Iterable[InputLayer],Iterable[OutputLayer])

  class BackPropagationExample(val inputLayers:Iterable[InputLayer], val outputLayers:Iterable[OutputLayer], val scale:Double = 1.0) extends Example {
    val computationSeq = calculateComputationSeq(inputLayers) //cache that, so it doesnt have to be computed all the time
    override def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
      _forwardPropagateInput(computationSeq,inputLayers)
      val gradients = _backPropagateOutputGradient(computationSeq,gradient,scale)
      if(value != null)
        value.accumulate(outputLayers.foldLeft(0.0)(_ + _.lastObjective) * scale)
    }
    //sample is number of samples per weights
    def checkGradient(gradient:WeightsMap = null, sample:Int = -1) = {
      var check = true
      val g = {
        if (gradient == null) {
          _forwardPropagateInput(computationSeq,inputLayers)
          _backPropagateOutputGradient(computationSeq)
        } else gradient
      }

      val epsilon: Double = 1e-5
      val diffThreshold: Double = 0.01
      val diffPctThreshold: Double = 0.1

      g.keys.foreach(w => {
        g(w).foreachActiveElement((i,calcDeriv) => {
          if(sample < 0 || rng.nextInt(w.value.length) < sample) {
            val v = w.value(i)
            w.value.update(i, v + epsilon)
            _forwardPropagateInput(computationSeq, inputLayers)
            val e1 = totalObjective(outputLayers)
            w.value.update(i, v - epsilon)
            _forwardPropagateInput(computationSeq, inputLayers)
            val e2 = totalObjective(outputLayers)
            val appDeriv: Double = (e1 - e2) / (2 * epsilon)
            val diff = math.abs(appDeriv - calcDeriv)
            val pct = diff / Math.min(Math.abs(appDeriv), Math.abs(calcDeriv))
            w.value.update(i, v)
            if (diff > diffThreshold && pct > diffPctThreshold) {
              check = false
              logger.warn(s"gradient check failed with difference: $diff > $diffThreshold")
            }
          }
        })
      })
      check
    }
  }
}

//Example feed-forward model
class BasicFeedForwardNeuralNetwork(structure:Array[(Int,ActivationFunction)],objectiveFunction:MultivariateOptimizableObjective[Tensor1] = new SquaredMultivariate) extends FeedForwardNeuralNetworkModel {
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

  class OutputLayer extends BasicOutputNeuralNetworkLayer(NNUtils.newDense(structure.last._1),structure(structure.length-1)._2,objectiveFunction) with Layer {
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
}
