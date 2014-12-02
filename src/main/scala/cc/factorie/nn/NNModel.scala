package cc.factorie.nn

import cc.factorie.la.{WeightsMapAccumulator, Tensor1}
import cc.factorie.model._
import cc.factorie.optimize.{Example, MultivariateOptimizableObjective}
import cc.factorie.optimize.OptimizableObjectives.SquaredMultivariate
import cc.factorie.util.{FastLogging, DoubleAccumulator}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random


trait NNModel extends Parameters {
  implicit val rng = new Random(9363628)
  type Connection = NNWeights#Connection //easier access
  def connections(layer: NNLayer): Iterable[Connection]
}


//Usage: first createInputOutputLayerPairs, and train using any trainer and optimizer with BackpropagationExample, which takes as input an Iterable of InputLayers.
// this mixin can handle any kind of network structure that is a DAG
trait FeedForwardNNModel extends NNModel with FastLogging {
  type InputLayer <: InputNNLayer
  type OutputLayer <: OutputNNLayer
  type Input
  type Output

  //represents ordered execution/update sequence of factors/layers, from input to output
  type OrderedConnections = Seq[(Iterable[Connection],Iterable[NNLayer])]

  def totalObjective(outputLayers:Iterable[OutputLayer]) = {
    outputLayers.map(o => {
      o.objectiveGradient
      o.lastObjective
    }).sum
  }

  def forwardPropagateInput(inputLayers:Iterable[InputLayer]) = {
    val orderedConnections = calculateComputationSeq(inputLayers)
    _forwardPropagateInput(orderedConnections,inputLayers)
  }

  protected def _forwardPropagateInput(orderedConnections:OrderedConnections, inputLayers:Iterable[InputLayer]) = {
    orderedConnections.foreach(_._2.foreach(_.zeroInput()))
    orderedConnections.foreach(cs => {
      cs._1.foreach { _.forwardPropagate() }
      cs._2.foreach(_.updateActivation())
    })
  }

  //returns gradient on the error function for all weights of this model
  def backPropagateOutputGradient(inputLayers:Iterable[InputLayer]):WeightsMap= {
    val orderedConnections = calculateComputationSeq(inputLayers)
    _backPropagateOutputGradient(orderedConnections)
  }

  protected def _backPropagateOutputGradient(orderedConnections: OrderedConnections): WeightsMap = {
    val map = new WeightsMap(key => key.value.blankCopy)
    orderedConnections.foreach(_._2.withFilter(!_.isInstanceOf[OutputLayer]).foreach(_.zeroObjectiveGradient()))
    val gradients = orderedConnections.reverseIterator.flatMap(cs => {
      //multiply accumulated gradient with derivative of activation
      cs._2.foreach(_.updateObjectiveGradient())
      //backpropagate gradient
      val e = cs._1.flatMap(f => {val grad = f.backPropagateGradient; if(grad != null) Some(f.family.weights -> f.backPropagateGradient) else None})
      e
    })
    gradients.foreach{
      case (weights,gradient) =>
        map(weights) += gradient
    }
    map
  }

  protected def _backPropagateOutputGradient(orderedConnections: OrderedConnections,accumulator:WeightsMapAccumulator,scale:Double=1.0) = {
    orderedConnections.foreach(_._2.withFilter(!_.isInstanceOf[OutputLayer]).foreach(_.zeroObjectiveGradient()))
    val gradients = orderedConnections.reverseIterator.flatMap(cs => {
      //multiply accumulated gradient with derivative of activation
      cs._2.foreach(_.updateObjectiveGradient())
      //backpropagate gradient
      val e = cs._1.map(f => f.family.weights -> f.backPropagateGradient)
      e
    })
    gradients.foreach {
      case (weights,gradient) =>
        if(scale ==1.0) accumulator.accumulate(weights, gradient,scale)
        else accumulator.accumulate(weights, gradient)
    }
  }

  def forwardAndBackPropagateOutputGradient(inputLayers:Iterable[InputLayer]):WeightsMap = {
    val orderedConnections = calculateComputationSeq(inputLayers)
    _forwardPropagateInput(orderedConnections,inputLayers)
    _backPropagateOutputGradient(orderedConnections)
  }

  //computes the DAG starting from the input layers and returns a seq of independent factors and layers which inputs comes only from factors up to this point
  //Override this if there is a more efficient way of computing this (e.g.: see BasicFeedForwardNeuralNetwork)
  def calculateComputationSeq(inputLayers:Iterable[InputLayer]): OrderedConnections = {
    var currentLayers = mutable.HashSet[NNLayer]() ++ inputLayers
    val updatedLayers = mutable.HashSet[NNLayer]() ++ inputLayers
    val connectionSeq = ArrayBuffer[(ArrayBuffer[Connection],Iterable[NNLayer])]()

    val updatedInputLayers = mutable.Map[Connection,(Int,Int)]()
    val updatedInputConnections = mutable.Map[NNLayer,(Int,Int)]()

    while(currentLayers.nonEmpty) {
      val nextConnections = ArrayBuffer[Connection]()
      val nextLayers = mutable.HashSet[NNLayer]()
      currentLayers.foreach(l => outputConnections(l).foreach(f => {
        val outLayers = f.outputLayers
        val (nrInputs,totalNrInputs) = updatedInputLayers.getOrElseUpdate(f,(0,f.inputLayers.size))
        if (nrInputs == totalNrInputs - 1) {
          nextConnections += f
          //update number of incoming activated factors for each output layer of this factor; if full, add it to nextConnections
          outLayers.foreach(l => {
            val (ct,totalCt) =
              updatedInputConnections.getOrElseUpdate(l, {val inConnections = inputConnections(l); (inConnections.count(_.numVariables == 1),inConnections.size) }) //initialize with number of biases
            if(ct == totalCt - 1) //this layer got full input, add it as next layer
              nextLayers += l
            else //update incoming count for this layer
              updatedInputConnections += l -> (ct+1,totalCt)
          } )
        } else
          updatedInputLayers(f) = (nrInputs+1,totalNrInputs)
      }))
      currentLayers = nextLayers
      updatedLayers ++= currentLayers

      //Add biases
      currentLayers.foreach(l => nextConnections ++= inputConnections(l).filter(_.numVariables == 1))
      if(nextConnections.nonEmpty)
        connectionSeq += ((nextConnections,nextLayers))
    }
    connectionSeq
  }

  def inputConnections(layer: NNLayer):Iterable[Connection]
  def outputConnections(layer: NNLayer):Iterable[Connection]

  def connections(layer: NNLayer): Iterable[Connection] = inputConnections(layer) ++ outputConnections(layer)

  //creates network input and corresponding labeled output (for training). It is possible that the network architecture has many input and output layers, depending on its architecture
  def createNetwork(input:Input,output:Output):(Iterable[InputLayer],Iterable[OutputLayer])
  //creates network only based on input. Output is usually not labeled here (for prediction)
  def createNetwork(input:Input):(Iterable[InputLayer],Iterable[OutputLayer])
  class BackPropagationExample(val inputLayers:Iterable[InputLayer], val outputLayers:Iterable[OutputLayer], val scale:Double = 1.0) extends Example {
    val computationSeq = calculateComputationSeq(inputLayers) //cache that, so it doesnt have to be computed all the time
    override def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
      _forwardPropagateInput(computationSeq,inputLayers)
      _backPropagateOutputGradient(computationSeq,gradient,scale)
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

trait ItemizedFeedForwardNN extends FeedForwardNNModel {
  private val inConnections = mutable.Map[NNLayer,List[Connection]]()
  private val outConnections = mutable.Map[NNLayer,List[Connection]]()
  def newConnection(weights:NNWeights,layers:NNLayer*) = {
    val c = weights.newConnection(layers)
    c.outputLayers.map(l => inConnections += l -> (c :: inConnections.getOrElse(l,List[Connection]())))
    c.inputLayers.map(l => outConnections += l -> (c :: outConnections.getOrElse(l,List[Connection]())))
    c
  }
  override def inputConnections(layer: NNLayer): Iterable[Connection] = inConnections.getOrElse(layer,List[Connection]())
  override def outputConnections(layer: NNLayer): Iterable[Connection] = outConnections.getOrElse(layer,List[Connection]())
}

//Example feed-forward model
class BasicFeedForwardNN(structure:Array[(Int,ActivationFunction)],objectiveFunction:MultivariateOptimizableObjective[Tensor1] = new SquaredMultivariate) extends FeedForwardNNModel {
  trait InputLayer extends InnerLayer with InputNNLayer
  type Input = Tensor1
  type Output = Tensor1

  val weights:Array[BasicLayerToLayerWeights[Layer,Layer]] = (0 until structure.length-1).map { case i =>
    val in = structure(i)._1
    val out = structure(i+1)._1
    new BasicLayerToLayerWeights[Layer,Layer] {
      override val weights: Weights2 = Weights(NNUtils.fillDense(in,out)((_,_) => Random.nextGaussian()/10))
      override type FamilyType = BasicLayerToLayerWeights[Layer,Layer]
     }
  }.toArray

  val biases:Array[Bias[Layer]] = (1 until structure.length).map { case i =>
    val out = structure(i)._1
    new Bias[Layer] {
      override val weights: Weights1 = Weights(NNUtils.fillDense(out)(_ => Random.nextGaussian()/10))
      override type FamilyType = Bias[Layer]
    }
  }.toArray

  trait Layer extends NNLayer {
    def index:Int
    var nextConnection:BasicLayerToLayerWeights[Layer,Layer]#ConnectionType = null.asInstanceOf[BasicLayerToLayerWeights[Layer,Layer]#ConnectionType]
    var prevConnection:BasicLayerToLayerWeights[Layer,Layer]#ConnectionType = null.asInstanceOf[BasicLayerToLayerWeights[Layer,Layer]#ConnectionType]
    lazy val bias:Bias[Layer]#ConnectionType = {
      if (index > 0) {
        val bias = biases(index-1)
        new bias.Connection(this)
      }
      else null
    }
  }
  class InnerLayer(override val index:Int) extends BasicNNLayer(structure(index)._1,structure(index)._2) with Layer {
    nextConnection = {
      if (index < structure.length - 2){
        val w = weights(index)
        new w.Connection(this, new InnerLayer(index + 1))
      }
      else null
    }
    if(nextConnection != null)
      nextConnection._2.prevConnection = nextConnection
  }

  class OutputLayer extends BasicOutputNNLayer(NNUtils.newDense(structure.last._1),structure(structure.length-1)._2,objectiveFunction) with Layer {
    val index = structure.length-1
  }

  def createNetwork(input:Tensor1, output:Tensor1):(Iterable[InputLayer],Iterable[OutputLayer]) = {
    val inputLayer = new InnerLayer(0) with InputLayer
    inputLayer.set(input)(null)
    var l:Layer = inputLayer
    while(l.nextConnection != null)
      l = l.nextConnection._2
    val out = new OutputLayer with LabeledNNLayer {
      override val target = new BasicTargetNNLayer(output,this)
    }
    val c = weights.last.newConnection(l, out)
    l.nextConnection = c
    out.prevConnection = c
    (Iterable(inputLayer),Iterable(l.asInstanceOf[OutputLayer]))
  }

  override def createNetwork(input: Input): (Iterable[InputLayer], Iterable[OutputLayer]) = {
    val inputLayer = new InnerLayer(0) with InputLayer
    inputLayer.set(input)(null)
    var l:Layer = inputLayer
    while(l.nextConnection != null)
      l = l.nextConnection._2
    val out = new OutputLayer //create no labeled var here
    val c = weights.last.newConnection(l, out)
    l.nextConnection = c
    out.prevConnection = c
    (Iterable(inputLayer),Iterable(l.asInstanceOf[OutputLayer]))
  }

  override def inputConnections(variable: NNLayer): Iterable[Connection] = variable match {
    case v:Layer =>
      var cs = List[Connection]()
      if(v.prevConnection != null)
        cs ::= v.prevConnection.asInstanceOf[Connection]
      if(v.bias!=null)
        cs ::= v.bias.asInstanceOf[Connection]
      cs
  }
  override def outputConnections(variable: NNLayer): Iterable[Connection] = variable match {
    case v:Layer =>
      if(v.nextConnection != null)
        List(v.nextConnection)
      else
        List[Connection]()
  }
}
