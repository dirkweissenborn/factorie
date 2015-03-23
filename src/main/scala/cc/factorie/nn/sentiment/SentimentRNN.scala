package cc.factorie.nn.sentiment

import java.io.File

import cc.factorie.la._
import cc.factorie.model._
import cc.factorie.nn._
import cc.factorie.nn.weights._
import cc.factorie.optimize._
import cc.factorie.util.FastLogging
import cc.factorie.variable.CategoricalDomain
import scala.util.Random

class SentimentRNN(dim:Int, tokenDomain:CategoricalDomain[String], withTensors:Boolean = true, val numLabels:Int = 5,
                   activation:ActivationFunction = ActivationFunction.HardTanh) extends FeedForwardNNModel {
  assert(tokenDomain.frozen)

  private val rand = new Random(6393854)

  // #### First define layers of this network, input and output ####
  override type Output = Any //included in input
  override type Input = SentimentPTree
  var objectiveF = labelToTensorObjective(OptimizableObjectives.hingeMulticlass)

  case class OutputUnit(label:Int, inputLayer:Unit) extends BasicOutputNNUnit({val t = TensorUtils.newDense(numLabels);t.+=(label,1.0);t},objectiveFunction = objectiveF) with LabeledNNUnit {
    override def toString: String = s"Sentiment: $label"
    override val target = new BasicTargetNNUnit(new SingletonTensor1(numLabels,label,1.0), this)
    newConnection(outBias,this)
  }
  case class InputUnit(token:String) extends OneHotNNUnit(new SingletonBinaryTensor1(tokenDomain.size, tokenDomain.index(token))) {
    override def toString: String = token
  }
  case class Unit(node:SentimentPNode, childLayer1:Unit = null, childLayer2:Unit = null, isRoot:Boolean = false)
    extends BasicNNUnit(dim, activation) {

    if(childLayer1 != null) {
      newConnection(childWeights1,childLayer1,this)
      newConnection(childWeights2,childLayer2,this)
      if(withTensors) newConnection(tensorWeights,childLayer1,childLayer2,this)
    }
    val output = OutputUnit(node.score,this)
    newConnection(outputWeights, this, output)
    val inputLayer:InputUnit = if(node.label!="") InputUnit(node.label) else null
    if(inputLayer != null) newConnection(embeddings,inputLayer,this)
    newConnection(bias,this)

    private lazy val treeString ={
      "("+node.score+" "+ {if(childLayer1==null)node.label else childLayer1.toString()} + {if(childLayer2!=null) " "+childLayer2.toString() else "" } + ")"
    }
    override def toString: String = treeString
  }
  //define weight families
  val childWeights1 = new FullConnection[Unit,Unit] {
    override lazy val weights: Weights2 = Weights(TensorUtils.fillDense(dim,dim)((i,j) => { if(i==j)1.0 else 0.0 }+(rand.nextDouble()-0.5)/math.sqrt(dim)))
  }
  val childWeights2 = new FullConnection[Unit,Unit] {
    override lazy val weights: Weights2 = Weights(TensorUtils.fillDense(dim,dim)((i,j) => { if(i==j)1.0 else 0.0 }+(rand.nextDouble()-0.5)/math.sqrt(dim)))
  }
  val tensorWeights = new NeuralTensorConnection[Unit,Unit,Unit] {
    override lazy val weights: Weights3 = Weights(TensorUtils.fillDense(dim,dim,dim)((i,j,k) => (rand.nextDouble()-0.5)/(2*dim)))
  }
  val bias = new Bias[Unit] {
    override lazy val weights: Weights1 = Weights(TensorUtils.newDense(dim))
  }
  val outBias = new Bias[OutputUnit] {
    override lazy val weights: Weights1 = Weights(TensorUtils.newDense(numLabels))
  }
  val outputWeights = new FullConnection[Unit,OutputUnit] {
    override lazy val weights: Weights2 = Weights(TensorUtils.fillDense(dim,numLabels)((_,_) => 2.0*(rand.nextDouble()-0.5)/math.sqrt(dim)))
  }
  val embeddings = new FullConnection[OneHotNNUnit, Unit] {
    //override lazy val weights: Weights2 = Weights(new RowVectorMatrix(tokenDomain.size,dim, d => NNUtils.fillDense(d)(_ => rand.nextGaussian())).init())
    override lazy val weights: Weights2 = Weights(TensorUtils.fillDense(tokenDomain.size,dim)((_,_) => rand.nextDouble()/10000.0)) //faster than above
  }

  //Now define how to create a network given the input
  //creates network input and corresponding output. It is possible that the network architecture has many input and output layers, depending on its architecture
  override def createNetwork(input: Input, output: Output = None) = createNetwork(input)

  override def createNetwork(input: Input) = {
    val layers = new Array[Unit](input.nodes.length)
    input.nodes.foreach(n => {
      val isRoot = n ==  input.nodes.last
      val layer = if(n.label!="") new Unit(n,isRoot=isRoot) else Unit(n,layers(n.c1),layers(n.c2),isRoot)
      layers(n.id) = layer
    })
    new Network(layers.withFilter(_.inputLayer != null).map(_.inputLayer),layers.map(_.output))
  }
}

class TernarySentimentRNN(dim:Int, tokenDomain:CategoricalDomain[String], withTensors:Boolean = true, activation:ActivationFunction = ActivationFunction.Tanh)
  extends SentimentRNN(dim,tokenDomain,withTensors, 3,activation) {
  override def createNetwork(input: Input) = {
    val layers = new Array[Unit](input.nodes.length)
    input.nodes.foreach(n => {
      if(n.score > 2) n.score = 2
      else if(n.score == 2) n.score = 1
      else if(n.score < 2) n.score = 0
      val isRoot = n ==  input.nodes.last
      val layer = if(n.label!="") new Unit(n,isRoot=isRoot) else Unit(n,layers(n.c1),layers(n.c2),isRoot)
      layers(n.id) = layer
    })
    new Network(layers.withFilter(_.inputLayer != null).map(_.inputLayer),layers.map(_.output))
  }
}

object SentimentRNN extends FastLogging {
  def train(train:File,test:File,dev:File) {
    TensorUtils.setTensorImplementation(TensorUtils.JBLAS)
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
        val net = model.createNetwork(t)
        new model.BackPropagationExample(net,1.0/batchSize)//,rand.nextDouble() >= 0.99) //gradient check at every 100th example, expensive
      }.toList
    }
    val trainExamples = createExamples(trainTrees)
    val testExamples = testTrees.map(t => model.createNetwork(t))
    val devExamples = devTrees.map(t => model.createNetwork(t)).toSeq

    val zeroLabel = model.numLabels/2
    def printEvaluation(examples:Seq[model.Network], name:String) {
      val size = examples.view.map(_.outputUnits.size).sum.toDouble
      val objective = examples.view.map(e => e.totalObjective).sum / size
      var binaryAll = 0.0
      var binaryAllTotal = 0.0
      var binaryRoot = 0.0
      var binaryRootTotal = 0.0

      val fineGrainedAll = examples.view.map(_.outputUnits.count(o => {
        val t = o.target.value.maxIndex
        val a = o.value.maxIndex
        if(t != zeroLabel) binaryAllTotal+=1
        if(t > zeroLabel && a > zeroLabel) binaryAll += 1
        if(t < zeroLabel && a < zeroLabel) binaryAll += 1
        t == a
      })).sum / size

      val fineGrainedRoot = examples.count(e => {
        val r = e.outputUnits.find(o => o.inputLayer.isRoot).get
        val t = r.target.value.maxIndex
        val a = r.value.maxIndex
        if(t != zeroLabel) binaryRootTotal+=1
        if(t > zeroLabel && a > zeroLabel) binaryRoot += 1
        if(t < zeroLabel && a < zeroLabel) binaryRoot += 1
        t == a
      }) / examples.size.toDouble
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
      printEvaluation(trainExamples.map(_.network),"Train")

      devExamples.foreach(_.forwardPropagateInput())
      printEvaluation(devExamples,"Dev")

      testExamples.foreach(_.forwardPropagateInput())
      printEvaluation(testExamples,"Test")
    }
  }

  def main (args: Array[String]) {
    val Array(training,test,dev) = args
    train(new File(training),new File(test),new File(dev))
  }
}
