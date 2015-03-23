package cc.factorie.nn


import cc.factorie.la._
import cc.factorie.model._
import cc.factorie.nn.weights._
import cc.factorie.optimize.{Example, MultivariateOptimizableObjective}
import cc.factorie.optimize.OptimizableObjectives.SquaredMultivariate
import cc.factorie.util.{FastLogging, DoubleAccumulator}
import com.mongodb.util.IdentitySet
import scala.collection.mutable
import scala.collection.mutable.{ListBuffer, ArrayBuffer}
import scala.util.Random
import scala.collection.JavaConversions._


trait NNModel extends Parameters {
  implicit val rng = new Random(9363628)
  type Connection = NNConnection#Layer //easier access
  def connections(layer: NNUnit): Iterable[Connection]
}

//Usage: first createInputOutputLayerPairs, and train using any trainer and optimizer with BackpropagationExample, which takes as input an Iterable of InputLayers.
// this mixin can handle any kind of network structure that is a DAG
trait FeedForwardNNModel extends NNModel with FastLogging {
  type InputUnit <: InputNNUnit
  type OutputUnit <: OutputNNUnit
  type Input
  type Output

  //represents ordered execution/update sequence of connections/layers, from input to output
  type OrderedConnections = Seq[(Iterable[Connection],Iterable[NNUnit])]

  class Network(val inputUnits:Iterable[InputUnit],val outputUnits:Iterable[OutputUnit]) {
    def computationSeq:OrderedConnections = _computationSeq
    
    private val _computationSeq = ArrayBuffer[(Iterable[Connection],Iterable[NNUnit])]()
    recalculateComputationSeq()
    
    def recalculateComputationSeq() = {
      var currentUnits:Iterable[NNUnit] = inputUnits
      _computationSeq.clear()

      val updatedInputConnections = mutable.AnyRefMap[Connection,(Int,Int)]()
      //connection -> (#currently activated input units,#total inputs), if both are equal the connection can be activated
      val updatedInputUnits = mutable.AnyRefMap[NNUnit,(Int,Int)]()
      //unit -> (#currently activated input connections,#total inputs), if both are equal the units can be activated

      lazy val recurrentUnitTime = mutable.AnyRefMap[RecurrentNNUnit,Int]() withDefaultValue 0
      //connections to recurrent units need to be active at every timestep 
      lazy val recConnections = mutable.AnyRefMap[RecurrentNNUnit,IdentitySet[NNConnection#Layer]]()
      
      //keep track of added recurrent units, because they would get added multiple times otherwise
      lazy val addedUnits = new IdentitySet[NNUnit]()
      while(currentUnits.nonEmpty) {
        val nextConnections = new ListBuffer[Connection]()
        val nextUnits = new ListBuffer[NNUnit]()
        addedUnits.clear()
        currentUnits.foreach(unit => outputConnections(unit).foreach(c => {
          if(!c.connectionType.isInstanceOf[SpecificTimeConnection] || (unit match {
            //check whether connection can be made
            //TODO this is not nice
            case r:RecurrentNNUnit =>
              c.connectionType.asInstanceOf[SpecificTimeConnection].time == recurrentUnitTime(r)
            case _ => true
          }) ) {
            val outLayers = c.outputUnits
            val (nrInputs, totalNrInputs) = updatedInputConnections.getOrElse(c, (0, c.inputUnits.size))
            if (nrInputs == totalNrInputs - 1) {
              //update number of incoming activated connections for each output layer of this connection; if full, add it to nextConnections
              if(outLayers.nonEmpty)
                outLayers.foreach {
                  case r: RecurrentNNUnit =>
                    recConnections.getOrElseUpdate(r, new IdentitySet[NNConnection#Layer]()).add(c)
                    if(recurrentUnitTime(r) < r.maxTime && !addedUnits.contains(r)) {
                      nextConnections += c
                      addedUnits.add(r)
                      nextUnits += r
                    }
                  case u: NNUnit =>
                    nextConnections += c
                    val (ct, totalCt) =
                      updatedInputUnits.getOrElse(u, {
                        val inConnections = inputConnections(u)
                        (inConnections.count(_.numUnits == 1), inConnections.size)
                      }) //initialize with number of biases
                    if (ct == totalCt - 1) //this layer got full input, add it as next layer
                      nextUnits += u
                    else //update activated connections count for this layer
                      updatedInputUnits += u ->(ct + 1, totalCt)
                }
              else nextConnections.add(c)
            } else //update activated input layers count for this connections
              updatedInputConnections(c) = (nrInputs + 1, totalNrInputs)
          }
        }))
        currentUnits = nextUnits.result()
        currentUnits.foreach {
          case r:RecurrentNNUnit =>
            recurrentUnitTime(r) += 1
            nextConnections ++= recConnections(r)
          case _ => //
        }

        //add biases of nextUnits
        currentUnits.foreach(l => inputConnections(l).withFilter(_.numUnits == 1).foreach(nextConnections.add))
        if(nextConnections.nonEmpty)
          _computationSeq += ((nextConnections.result(),currentUnits))
      }
    }
    
    def foreachUnit(f:(NNUnit)=>Unit) = {
      computationSeq.foreach(p => inputUnits.foreach(f))
      computationSeq.foreach(p => p._2.foreach(f))
    }
    def foreachConnection(f:(Connection)=>Unit) = {
      computationSeq.foreach(p => p._1.foreach(f))
    }
    
    def foreachRecurrentUnit(f:(RecurrentNNUnit)=>Unit) = {
      foreachUnit {
        case r:RecurrentNNUnit => f(r)
        case _ => //Nothing        
      }
    }

    def totalObjective = {
      outputUnits.view.map(o => {
        //o.objectiveGradient
        o.lastObjective
      }).sum
    }

    def forwardPropagateInput() = {
      foreachUnit(_.reset())
      foreachConnection(_.reset())
      computationSeq.foreach(_._2.foreach(_.zeroInput()))
      computationSeq.foreach(cs => {
        cs._2.foreach {
          case r: RecurrentNNUnit => r.stepForwardInTime()
          case _ => //
        }
        cs._1.withFilter(!_.disabled).foreach { _.forwardPropagate() }
        cs._2.foreach(_.updateActivation())
      })
    }

    //returns gradient on the error function for all connection of this model
    def backPropagateOutputGradient:WeightsMap= {
      val map = new WeightsMap(key => key.value.blankCopy)
      backPropagateOutputGradient(new LocalWeightsMapAccumulator(map))
      map
    }

    def backPropagateOutputGradient(accumulator:WeightsMapAccumulator,scale:Double=1.0) = {
      computationSeq.foreach(_._2.foreach(_.zeroObjectiveGradient()))
      val updatedGrads = new IdentitySet[NNUnit]()
      //updatedGrads.addAll(outputUnits)
      val gradients = _computationSeq.reverseIterator.flatMap(cs => {
        //multiply accumulated gradient with derivative of activation
        cs._2.withFilter(_.gradientNeedsUpdate).foreach(u => {
          u.updateObjectiveGradient()
          updatedGrads.add(u)
        })
        cs._2.foreach {
          case r: RecurrentNNUnit => r.stepBackwardInTime()
          case u: NNUnit=> //nothing to do
        }
        //backpropagate gradient
        val e = cs._1.withFilter(c => (c.outputUnits.isEmpty ||
          c.outputUnits.exists(updatedGrads.contains)) && !c.disabled)
          .flatMap(c => {
          val grad = c.backPropagateGradient
          if (grad != null) Some(c.connectionType.weights -> grad)
          else None
        })
        e
      })
      gradients.foreach {
        case (weight, gradient) =>
          if (scale == 1.0) accumulator.accumulate(weight, gradient)
          else accumulator.accumulate(weight, gradient, scale)
      }
      //also accumulate local weights of units, e.g., word embedding units
      foreachUnit {
        case u:AccumulatableNNUnit =>
          if(updatedGrads.contains(u) || u.gradientNeedsUpdate)
            u.weightAndGradient.foreach(t => accumulator.accumulate(t._1,t._2,t._3*scale))
        case _ => //Nothing to do
      }
    }

    def forwardAndBackPropagateOutputGradient():WeightsMap = {
      forwardPropagateInput()
      backPropagateOutputGradient
    }
  }

  def newConnection(connection:NNConnection,layers:Seq[NNUnit]):Connection = {
    val c = connection.newConnection(layers)
    c.outputUnits.map(l => l.addInConnection(c))
    c.inputUnits.map(l => l.addOutConnection(c))
    c
  }
  def newConnection[L1 <: NNUnit](connection:NNConnection1[L1],l1:L1):Connection = newConnection(connection,Seq(l1))
  def newConnection[L1 <: NNUnit,L2 <: NNUnit](connection:NNConnection2[L1,L2],l1:L1,l2:L2):Connection = newConnection(connection,Seq(l1,l2))
  def newConnection[L1 <: NNUnit,L2 <: NNUnit,L3 <: NNUnit](connection:NNConnection3[L1,L2,L3],l1:L1,l2:L2,l3:L3):Connection = newConnection(connection,Seq(l1,l2,l3))
  def newConnection[L1 <: NNUnit,L2 <: NNUnit,L3 <: NNUnit,L4 <: NNUnit](connection:NNConnection4[L1,L2,L3,L4],l1:L1,l2:L2,l3:L3,l4:L4):Connection = newConnection(connection,Seq(l1,l2,l3,l4))

  def inputConnections(layer: NNUnit):Iterable[Connection] = layer.inConnections
  def outputConnections(layer: NNUnit):Iterable[Connection] = layer.outConnections

  def connections(layer: NNUnit): Iterable[Connection] = inputConnections(layer) ++ outputConnections(layer)

  //creates network input and corresponding labeled output (for training). It is possible that the network architecture has many input and output layers, depending on its architecture
  def createNetwork(input:Input,output:Output):Network
  //creates network only based on input. Output is usually not labeled here (for prediction)
  def createNetwork(input:Input):Network

  class BackPropagationExample(val network:Network, val scale:Double = 1.0) extends Example {
    //This is probably training so set recurrent units to track their history
    network.foreachRecurrentUnit(_.keepHistory = true)
    
    override def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
      network.forwardPropagateInput()
      network.backPropagateOutputGradient(gradient,scale)
      if(value != null)
        value.accumulate(network.outputUnits.foldLeft(0.0)(_ + _.lastObjective) * scale)
    }
    //sample is number of samples per connection
    def checkGradient(gradient:WeightsMap = null, sample:Int = -1) = {
      val g = {
        if (gradient == null) {
          network.forwardAndBackPropagateOutputGradient()
        } else gradient
      }

      val epsilon: Double = 1e-5
      val diffThreshold: Double = 0.01
      val diffPctThreshold: Double = 0.1
      
      g.keys.view.filter(_.value.numDimensions > 1).forall(w => {
        g(w).forallActiveElements((i,calcDeriv) => {
          if(sample < 0 || rng.nextInt(w.value.length) < sample) {
            val v = w.value(i)
            w.value.update(i, v + epsilon)
            network.forwardPropagateInput()
            val e1 = network.totalObjective
            w.value.update(i, v - epsilon)
            network.forwardPropagateInput()
            val e2 = network.totalObjective
            val appDeriv: Double = (e1 - e2) / (2 * epsilon)
            val diff = math.abs(appDeriv - calcDeriv)
            val pct = diff / Math.min(Math.abs(appDeriv), Math.abs(calcDeriv))
            w.value.update(i, v)
            if (diff > diffThreshold && pct > diffPctThreshold) {
              logger.warn(s"gradient check failed with difference: $diff > $diffThreshold")
              false
            } else true
          } else true
        })
      })
    }
  }
}

trait ConvNN extends FeedForwardNNModel {
  //horizontal convolution
  def convConnectHorizontal[L1 <: NNUnit,L2 <: NNUnit](units:IndexedSeq[L1],
                                                         convWeights:collection.Map[Int,_ <: NNConnection],
                                                         outUnits:IndexedSeq[L2]) = {
    assert(convWeights.forall(_._2.numOutputUnits == 1),
      "horizontal convolution only with single output connections possible")
    val diff = (outUnits.length - units.length)/2
    (0 until outUnits.length).foreach(l => {
      convWeights.foreach { case (w,connection) =>
        val lc = l-diff+w
        if(lc >= 0 && lc < units.length) {
          connection match {
            case connection:NNConnection2[_,_] => //usual case one input layer, one output layer
              newConnection(connection,Seq(units(lc),outUnits(l)))
            case connection:NNConnection3[_,_,_] =>
              if(lc + 1 <= units.length )
                newConnection(connection,Seq(units(lc),units(lc+1),outUnits(l)))
            case connection:NNConnection4[_,_,_,_] =>
              if(lc + 1 <= units.length && lc-1 >= 0)
                newConnection(connection,Seq(units(lc),units(lc+1),outUnits(l)))
          }
        }
      }
    })
  }
  
}

//Example feed-forward model
class BasicFeedForwardNN(structure:Array[(Int,ActivationFunction)],objectiveFunction:MultivariateOptimizableObjective[Tensor1] = new SquaredMultivariate) extends FeedForwardNNModel {
  trait InputUnit extends InnerFFUnit with InputNNUnit
  type Input = Tensor1
  type Output = Tensor1

  val connections:Array[FullConnection[FFUnit,FFUnit]] = (0 until structure.length-1).map { case i =>
    val in = structure(i)._1
    val out = structure(i+1)._1
    new FullConnection[FFUnit,FFUnit] {
      override val weights: Weights2 = Weights(TensorUtils.fillDense(in,out)((_,_) => Random.nextGaussian()/10))
      override type ConnectionType = FullConnection[FFUnit,FFUnit]
     }
  }.toArray

  val biases:Array[Bias[FFUnit]] = (1 until structure.length).map { case i =>
    val out = structure(i)._1
    new Bias[FFUnit] {
      override val weights: Weights1 = Weights(TensorUtils.fillDense(out)(_ => Random.nextGaussian()/10))
      override type ConnectionType = Bias[FFUnit]
    }
  }.toArray

  trait FFUnit extends NNUnit {
    def index:Int
    var next:FFUnit = null
    if(index > 0)
      newConnection(biases(index - 1),this)
  }
  class InnerFFUnit(override val index:Int) extends BasicNNUnit(structure(index)._1,structure(index)._2) with FFUnit {
    if(index < structure.length - 2) {
      next = new InnerFFUnit(index + 1)
      newConnection(connections(index), this, next)
    }
  }

  class OutputUnit extends BasicOutputNNUnit(TensorUtils.newDense(structure.last._1),structure(structure.length-1)._2,objectiveFunction)
    with FFUnit{
    val index = structure.length-1
  }

  def createNetwork(input:Tensor1, output:Tensor1):Network = {
    val inputUnits = new InnerFFUnit(0) with InputUnit
    inputUnits.set(input)(null)
    var l:FFUnit = inputUnits
    while(l.next != null)
      l = l.next
    val out = if(output != null) new OutputUnit with LabeledNNUnit {
      override val target = new BasicTargetNNUnit(output,this)
    } else new OutputUnit
    newConnection(connections.last, l, out)
    new Network(Iterable(inputUnits),Iterable(out))
  }
  override def createNetwork(input: Input)= createNetwork(input,null)
}
