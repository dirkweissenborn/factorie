package cc.factorie.nn.sentiment

import java.io.File

import cc.factorie.la._
import cc.factorie.model._
import cc.factorie.nn._
import cc.factorie.optimize._
import cc.factorie.util.FastLogging
import cc.factorie.variable.CategoricalDomain

import scala.collection.mutable
import scala.util.Random


class SentimentRNN(dim:Int, tokenDomain:CategoricalDomain[String], withTensors:Boolean = true, numLabels:Int = 5, activation:ActivationFunction = ActivationFunction.Tanh) extends FeedForwardNeuralNetworkModel {
  assert(tokenDomain.frozen)

  private val rand = new Random(Random.nextInt())

  // #### First define layers of this network, input and output ####
  override type Output = Any //included in input
  override type Input = SentimentPTree
  abstract class OutputLayer(label:Int) extends BasicOutputNeuralNetworkLayer({val t = NNUtils.newDense(numLabels);t.+=(label,1.0);t}) {
    val factor:outputWeights.FactorType
    override def toString: String = s"Sentiment: $label"
    val biasF = new outBias.Factor(this)
  }
  abstract class InputLayer(val token:String) extends OneHotLayer(new SingletonBinaryTensor1(tokenDomain.size, tokenDomain.index(token))) {
    val factor:embeddings.FactorType
    override def toString: String = token
  }
  class Layer(protected val node:SentimentPNode, childLayer1:Layer = null, childLayer2:Layer = null) extends BasicNeuralNetworkLayer(dim, activation) {
    self =>
    var parentFactor:BasicLayerToLayerWeightsFamily[Layer,Layer]#FactorType = null
    var parentTensorFactor:tensorWeights.FactorType = null
    val (childFactor1,childFactor2) =
      if(childLayer1 != null)
        (new childWeights1.Factor(childLayer1,this), new childWeights2.Factor(childLayer2,this))
      else (null,null)

    lazy val tensorFactor:tensorWeights.FactorType = if(withTensors && childLayer1 != null) {
      val t = new tensorWeights.Factor(childLayer1,childLayer2,this)
      childLayer1.parentTensorFactor = t
      childLayer2.parentTensorFactor = t
      t
    } else null

    if(childLayer1 != null) {
      childLayer1.parentFactor = childFactor1
      childLayer2.parentFactor = childFactor2
      childLayer2.parentTensorFactor = tensorFactor
    }
    val output = new OutputLayer(node.score) {
      override val factor = new outputWeights.Factor(self, this)
    }
    lazy val outputFactor = output.factor
    val inputLayer = if(node.label!="") new InputLayer(node.label) {
      override val factor: embeddings.FactorType = new embeddings.Factor(this,self)
    } else null
    lazy val inputFactor = inputLayer.factor

    private lazy val treeString ={
      "("+node.score+" "+ {if(childLayer1==null)node.label else childLayer1.toString()} + {if(childLayer2!=null) " "+childLayer2.toString() else "" } + ")"
    }

    val biasF = new bias.Factor(this)

    override def toString: String = treeString
  }
  //define weight families
  val childWeights1 = new BasicLayerToLayerWeightsFamily[Layer,Layer] {
    override lazy val weights: Weights2 = Weights(NNUtils.fillDense(dim,dim)((i,j) => { if(i==j)1.0 else 0.0 }+(rand.nextDouble()-0.5)/math.sqrt(dim)))
  }
  val childWeights2 = new BasicLayerToLayerWeightsFamily[Layer,Layer] {
    override lazy val weights: Weights2 = Weights(NNUtils.fillDense(dim,dim)((i,j) => { if(i==j)1.0 else 0.0 }+(rand.nextDouble()-0.5)/math.sqrt(dim)))
  }
  val tensorWeights = new NeuralTensorWeightsFamily[Layer,Layer,Layer] {
    override lazy val weights: Weights3 = Weights(NNUtils.fillDense(dim*2,dim*2,dim)((i,j,k) => (rand.nextDouble()-0.5)/(2*dim)))
  }
  val bias = new Bias[Layer] {
    override lazy val weights: Weights1 = Weights(NNUtils.newDense(dim))
  }
  val outBias = new Bias[OutputLayer] {
    override lazy val weights: Weights1 = Weights(NNUtils.newDense(numLabels))
  }
  val outputWeights = new BasicLayerToLayerWeightsFamily[Layer,OutputLayer] {
    override lazy val weights: Weights2 = Weights(NNUtils.fillDense(dim,numLabels)((_,_) => 2.0*(rand.nextDouble()-0.5)/math.sqrt(dim)))
  }
  val embeddings = new BasicLayerToLayerWeightsFamily[OneHotLayer, Layer] {
    override lazy val weights: Weights2 = Weights(NNUtils.fillDense(tokenDomain.size,dim)((_,_) => rand.nextGaussian()))
  }

  //Now define how to create a network given the input, and how to access all factors of a set of variables

  //creates network input and corresponding output. It is possible that the network architecture has many input and output layers, depending on its architecture
  override def createNetwork(input: Input, output: Output = None): (Iterable[InputLayer], Iterable[OutputLayer]) = {
    val layers = new Array[Layer](input.nodes.length)
    input.nodes.foreach(n => {
      val layer = if(n.label!="") new Layer(n) else new Layer(n,layers(n.c1),layers(n.c2))
      layers(n.id) = layer
    })
    (layers.withFilter(_.inputLayer != null).map(_.inputLayer),layers.map(_.output))
  }

  override def inputFactors(variable: NeuralNetworkLayer): Iterable[Factor] = variable match {
    case v:InputLayer => List()
    case v:Layer =>
      if(v.inputLayer!=null) Iterable(v.inputFactor,v.biasF)
      else if(withTensors) Iterable(v.childFactor1,v.childFactor2,v.biasF,v.tensorFactor)
      else Iterable(v.childFactor1,v.childFactor2,v.biasF)
    case v:OutputLayer => Iterable(v.factor,v.biasF)
  }

  override def outputFactors(variable: NeuralNetworkLayer): Iterable[Factor] = variable match {
    case v:InputLayer => List(v.factor)
    case v:Layer =>
      if(v.parentFactor ==null) Iterable(v.outputFactor)
      else if(withTensors) Iterable(v.parentFactor,v.outputFactor,v.parentTensorFactor)
      else Iterable(v.parentFactor,v.outputFactor)
    case v:OutputLayer => Iterable[Factor]()
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
    val model = new SentimentRNN(25,domain)

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
        if(t != 2) binaryAllTotal+=1
        if(t > 2 && a > 2) binaryAll += 1
        if(t < 2 && a < 2) binaryAll += 1
        t == a
      })).sum / examples.map(_.size).sum.toDouble

      val fineGrainedRoot = examples.count(e => {
        val r = e.find(_.factor._1.parentFactor == null).get
        val t = r.target.value.maxIndex
        val a = r.value.maxIndex
        if(t != 2) binaryRootTotal+=1
        if(t > 2 && a > 2) binaryRoot += 1
        if(t < 2 && a < 2) binaryRoot += 1
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
    while(iterations < 100) {
      iterations += 1
      val trainer = new ParallelBatchTrainer(
        model.parameters,
        maxIterations = -1,
        optimizer = new AdaGrad(rate = 0.01,delta = 0.001) with L2Regularization { variance = 1000.0 }) //Adagrad with L2-regularization

      trainExamples.grouped(batchSize).foreach(e => trainer.processExamples(e))
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
