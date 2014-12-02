package cc.factorie.nn.sentiment

import java.io.File

import cc.factorie.la._
import cc.factorie.model._
import cc.factorie.nn._
import cc.factorie.optimize._
import cc.factorie.util.FastLogging
import cc.factorie.variable.CategoricalDomain
import scala.util.Random

//Todo, can be tranformed to ItemizedFeedForwardNeuralNetwork, which handles connections... Removes complexity from the implementation
class SentimentRNN(dim:Int, tokenDomain:CategoricalDomain[String], withTensors:Boolean = true, val numLabels:Int = 5, activation:ActivationFunction = ActivationFunction.Tanh) extends FeedForwardNNModel {
  assert(tokenDomain.frozen)

  private val rand = new Random(Random.nextInt())

  // #### First define layers of this network, input and output ####
  override type Output = Any //included in input
  override type Input = SentimentPTree
  abstract class OutputLayer(label:Int) extends BasicOutputNNLayer({val t = NNUtils.newDense(numLabels);t.+=(label,1.0);t}) with LabeledNNLayer {
    val connection:outputWeights.ConnectionType
    override def toString: String = s"Sentiment: $label"
    val biasF = new outBias.Connection(this)
    override val target = new BasicTargetNNLayer(value.copy, this)
  }
  abstract class InputLayer(val token:String) extends OneHotNNLayer(new SingletonBinaryTensor1(tokenDomain.size, tokenDomain.index(token))) {
    val connection:embeddings.ConnectionType
    override def toString: String = token
  }
  class Layer(protected val node:SentimentPNode, childLayer1:Layer = null, childLayer2:Layer = null)
    extends BasicNNLayer(dim, activation) {
    self =>
    var parentConnection:BasicLayerToLayerWeights[Layer,Layer]#ConnectionType = null
    var parentTensorConnection:tensorWeights.ConnectionType = null
    val (childConnection1,childConnection2) =
      if(childLayer1 != null)
        (new childWeights1.Connection(childLayer1,this), new childWeights2.Connection(childLayer2,this))
      else (null,null)

    val tensorConnection:tensorWeights.ConnectionType = if(withTensors && childLayer1 != null) {
      val t = new tensorWeights.Connection(childLayer1,childLayer2,this)
      childLayer1.parentTensorConnection = t
      childLayer2.parentTensorConnection = t
      t
    } else null

    if(childLayer1 != null) {
      childLayer1.parentConnection = childConnection1
      childLayer2.parentConnection = childConnection2
    }
    val output = new OutputLayer(node.score) {
      override val connection = new outputWeights.Connection(self, this)
    }
    lazy val outputConnection = output.connection
    val inputLayer = if(node.label!="") new InputLayer(node.label) {
      override val connection: embeddings.ConnectionType = new embeddings.Connection(this,self)
    } else null
    lazy val inputConnection = inputLayer.connection

    private lazy val treeString ={
      "("+node.score+" "+ {if(childLayer1==null)node.label else childLayer1.toString()} + {if(childLayer2!=null) " "+childLayer2.toString() else "" } + ")"
    }

    val biasF = new bias.Connection(this)

    override def toString: String = treeString
  }
  //define weight families
  val childWeights1 = new BasicLayerToLayerWeights[Layer,Layer] {
    override lazy val weights: Weights2 = Weights(NNUtils.fillDense(dim,dim)((i,j) => { if(i==j)1.0 else 0.0 }+(rand.nextDouble()-0.5)/math.sqrt(dim)))
  }
  val childWeights2 = new BasicLayerToLayerWeights[Layer,Layer] {
    override lazy val weights: Weights2 = Weights(NNUtils.fillDense(dim,dim)((i,j) => { if(i==j)1.0 else 0.0 }+(rand.nextDouble()-0.5)/math.sqrt(dim)))
  }
  val tensorWeights = new NeuralTensorWeights[Layer,Layer,Layer] {
    override lazy val weights: Weights3 = Weights(NNUtils.fillDense(dim,dim,dim)((i,j,k) => (rand.nextDouble()-0.5)/(2*dim)))
  }
  val bias = new Bias[Layer] {
    override lazy val weights: Weights1 = Weights(NNUtils.newDense(dim))
  }
  val outBias = new Bias[OutputLayer] {
    override lazy val weights: Weights1 = Weights(NNUtils.newDense(numLabels))
  }
  val outputWeights = new BasicLayerToLayerWeights[Layer,OutputLayer] {
    override lazy val weights: Weights2 = Weights(NNUtils.fillDense(dim,numLabels)((_,_) => 2.0*(rand.nextDouble()-0.5)/math.sqrt(dim)))
  }
  val embeddings = new BasicLayerToLayerWeights[OneHotNNLayer, Layer] {
    //override lazy val weights: Weights2 = Weights(new RowVectorMatrix(tokenDomain.size,dim, d => NNUtils.fillDense(d)(_ => rand.nextGaussian())).init())
    override lazy val weights: Weights2 = Weights(NNUtils.fillDense(tokenDomain.size,dim)((_,_) => rand.nextDouble()/10000.0)) //faster than above
  }

  //Now define how to create a network given the input, and how to access all factors of a set of variables

  //creates network input and corresponding output. It is possible that the network architecture has many input and output layers, depending on its architecture
  override def createNetwork(input: Input, output: Output = None): (Iterable[InputLayer], Iterable[OutputLayer]) = createNetwork(input)

  override def createNetwork(input: Input): (Iterable[InputLayer], Iterable[OutputLayer]) = {
    val layers = new Array[Layer](input.nodes.length)
    input.nodes.foreach(n => {
      val layer = if(n.label!="") new Layer(n) else new Layer(n,layers(n.c1),layers(n.c2))
      layers(n.id) = layer
    })
    (layers.withFilter(_.inputLayer != null).map(_.inputLayer),layers.map(_.output))
  }

  override def inputConnections(variable: NNLayer): Iterable[Connection] = variable match {
    case v:InputLayer => List()
    case v:Layer =>
      if(v.inputLayer!=null) Iterable(v.inputConnection,v.biasF)
      else if(withTensors) Iterable(v.childConnection1,v.childConnection2,v.biasF,v.tensorConnection)
      else Iterable(v.childConnection1,v.childConnection2,v.biasF)
    case v:OutputLayer => Iterable(v.connection,v.biasF)
  }

  override def outputConnections(variable: NNLayer): Iterable[Connection] = variable match {
    case v:InputLayer => List(v.connection)
    case v:Layer =>
      if(v.parentConnection ==null) Iterable(v.outputConnection)
      else if(withTensors) Iterable(v.parentConnection,v.outputConnection,v.parentTensorConnection)
      else Iterable(v.parentConnection,v.outputConnection)
    case v:OutputLayer => Iterable[Connection]()
  }
}

class TernarySentimentRNN(dim:Int, tokenDomain:CategoricalDomain[String], withTensors:Boolean = true, activation:ActivationFunction = ActivationFunction.Tanh) extends SentimentRNN(dim,tokenDomain,withTensors, 3,activation) {
  override def createNetwork(input: Input): (Iterable[InputLayer], Iterable[OutputLayer]) = {
    val layers = new Array[Layer](input.nodes.length)
    input.nodes.foreach(n => {
      if(n.score > 2) n.score = 2
      else if(n.score == 2) n.score = 1
      else if(n.score < 2) n.score = 0
      val layer = if(n.label!="") new Layer(n) else new Layer(n,layers(n.c1),layers(n.c2))
      layers(n.id) = layer
    })
    (layers.withFilter(_.inputLayer != null).map(_.inputLayer),layers.map(_.output))
  }
}

object SentimentRNN extends FastLogging {
  def train(train:File,test:File,dev:File) {
    NNUtils.setTensorImplementation(NNUtils.EJML)
    val batchSize = 27

    val trainTrees = LoadPTB.sentimentPTBFromFile(train).toList
    val testTrees = LoadPTB.sentimentPTBFromFile(test).toList
    val devTrees = LoadPTB.sentimentPTBFromFile(dev).toList

    //Fill vocabulary
    val domain = new CategoricalDomain[String]()
    trainTrees.foreach(_.nodes.foreach(n => domain.index(n.label)))
    testTrees.foreach(_.nodes.foreach(n => domain.index(n.label)))
    devTrees.foreach(_.nodes.foreach(n => domain.index(n.label)))
    domain.freeze()

    //create RNN
    val model = new TernarySentimentRNN(25,domain,withTensors = true)

    //create examples
    def createExamples(trees:Iterable[SentimentPTree]) = {
      trees.map{ t =>
        val (in,out) =model.createNetwork(t)
        new model.BackPropagationExample(in,out,1.0/batchSize)//,rand.nextDouble() >= 0.99) //gradient check at every 100th example, expensive
      }.toList
    }
    val trainExamples = createExamples(trainTrees)
    val testExamples = testTrees.map(t => model.createNetwork(t))
    val devExamples = devTrees.map(t => model.createNetwork(t)).toSeq

    val zeroLabel = model.numLabels/2
    def printEvaluation(examples:Seq[Iterable[model.OutputLayer]], name:String) {
      val size = examples.size.toDouble
      val objective = examples.map(e => model.totalObjective(e)).sum / size
      var binaryAll = 0.0
      var binaryAllTotal = 0.0
      var binaryRoot = 0.0
      var binaryRootTotal = 0.0

      val fineGrainedAll = examples.map(_.count(e => {
        val t = e.target.value.maxIndex
        val a = e.value.maxIndex
        if(t != zeroLabel) binaryAllTotal+=1
        if(t > zeroLabel && a > zeroLabel) binaryAll += 1
        if(t < zeroLabel && a < zeroLabel) binaryAll += 1
        t == a
      })).sum / examples.map(_.size).sum.toDouble

      val fineGrainedRoot = examples.count(e => {
        val r = e.find(_.connection._1.parentConnection == null).get
        val t = r.target.value.maxIndex
        val a = r.value.maxIndex
        if(t != zeroLabel) binaryRootTotal+=1
        if(t > zeroLabel && a > zeroLabel) binaryRoot += 1
        if(t < zeroLabel && a < zeroLabel) binaryRoot += 1
        t == a
      }) / size
      logger.info(
        s"""####### Evaluation $name #############
           |Avg. objective: $objective
           |Finegrained all: $fineGrainedAll
           |Finegrained root: $fineGrainedRoot
           |Binary all: ${binaryAll / binaryAllTotal}
           |Binary root: ${binaryRoot / binaryRootTotal}
           |######################################
         """.stripMargin)
    }

    var iterations = 0
    //train the model
    //Gradient check, built
    trainExamples.head.checkGradient(sample = 10)
    while(iterations < 400) {
      iterations += 1
      val trainer = new ParallelBatchTrainer(
        model.parameters,
        maxIterations = -1,
        optimizer = new AdaGrad(rate = 0.01,delta = 0.001) with L2Regularization { variance = 1000.0 }) //Adagrad with L2-regularization

      Random.shuffle(trainExamples).grouped(batchSize).foreach(e => trainer.processExamples(e))
      printEvaluation(trainExamples.map(_.outputLayers),"Train")

      devExamples.foreach(e => model.forwardPropagateInput(e._1))
      printEvaluation(devExamples.map(_._2),"Dev")

      testExamples.foreach(e => model.forwardPropagateInput(e._1))
      printEvaluation(testExamples.map(_._2),"Test")
    }
  }

  def main (args: Array[String]) {
    val Array(training,test,dev) = args
    train(new File(training),new File(test),new File(dev))
  }
}
