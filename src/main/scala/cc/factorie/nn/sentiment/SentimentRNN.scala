package cc.factorie.nn.sentiment

import java.io.File

import cc.factorie.la._
import cc.factorie.model.{Weights1, Weights2}
import cc.factorie.nn._
import cc.factorie.optimize._
import cc.factorie.util.FastLogging
import cc.factorie.variable.CategoricalDomain

import scala.util.Random


class SentimentRNN(dim:Int, tokenDomain:CategoricalDomain[String],numLabels:Int = 5, activation:ActivationFunction = ActivationFunction.Tanh) extends FeedForwardNeuralNetworkModel {
  assert(tokenDomain.frozen)

  // #### First define layers of this network, input and output ####
  override type Output = Any //included in input
  override type Input = SentimentPTree
  abstract class OutputLayer(label:Int) extends BasicOutputNeuralNetworkLayer({val t = NNUtils.newDense(numLabels);t.+=(label,1.0);t}) {
    val factor:outputWeights.FactorType
    override def toString(): String = s"Sentiment: $label"
  }
  abstract class InputLayer(val token:String) extends OneHotLayer(new SingletonBinaryTensor1(tokenDomain.size, tokenDomain.index(token))) {
    val factor:embeddings.FactorType
    override def toString(): String = token
  }
  class Layer(protected val node:SentimentPNode, childLayer1:Layer = null, childLayer2:Layer = null) extends BasicNeuralNetworkLayer(dim, activation) {
    self =>
    var parentFactor:BasicLayerToLayerWeightsFamily[Layer,Layer]#FactorType = null
    val childFactor1  = if(childLayer1 != null) new childWeights1.Factor(childLayer1,this) else null
    val childFactor2  = if(childLayer2 != null) new childWeights2.Factor(childLayer2,this) else null
    if(childLayer1 != null) childLayer1.parentFactor = childFactor1
    if(childLayer2 != null) childLayer2.parentFactor = childFactor2
    val output = new OutputLayer(node.score) {
      override val factor = new outputWeights.Factor(self, this)
    }
    lazy val outputFactor = output.factor
    val inputLayer = if(node.label!="") new InputLayer(node.label) {
      override val factor: SentimentRNN.this.embeddings.FactorType = new embeddings.Factor(this,self)
    } else null
    lazy val inputFactor = inputLayer.factor

    private lazy val treeString ={
      "("+node.score+" "+ {if(childLayer1==null)node.label else childLayer1.toString()} + {if(childLayer2!=null) " "+childLayer2.toString() else "" } + ")"
    }

    override def toString(): String = treeString
  }
  //define weight families
  val childWeights1 = new BasicLayerToLayerWeightsFamily[Layer,Layer] {
    override val weights: Weights2 = Weights(NNUtils.fillDense(dim,dim)((_,_) => Random.nextDouble()/10000.0))
  }
  val childWeights2 = new BasicLayerToLayerWeightsFamily[Layer,Layer] {
    override val weights: Weights2 = Weights(NNUtils.fillDense(dim,dim)((_,_) => Random.nextDouble()/10000.0))
  }
  val bias = new Bias[Layer] {
    override val weights: Weights1 = Weights(NNUtils.fillDense(dim)((_) => Random.nextDouble()/10000.0))
  }
  val outputWeights = new BasicLayerToLayerWeightsFamily[Layer,OutputLayer] {
    override val weights: Weights2 = Weights(NNUtils.fillDense(dim,numLabels)((_,_) => Random.nextDouble()/10000.0))
  }
  val embeddings = new BasicLayerToLayerWeightsFamily[OneHotLayer, Layer] {
    override val weights: Weights2 = Weights(NNUtils.fillDense(tokenDomain.size,dim)((_,_) => Random.nextDouble()/10000.0))
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
      if(v.inputLayer!=null) Iterable(v.inputFactor)
      else Iterable(v.childFactor1,v.childFactor2)
    case v:OutputLayer => Iterable(v.factor)
  }

  override def outputFactors(variable: NeuralNetworkLayer): Iterable[Factor] = variable match {
    case v:InputLayer => List(v.factor)
    case v:Layer =>
      if(v.parentFactor ==null) Iterable(v.outputFactor)
      else Iterable(v.parentFactor,v.outputFactor)
    case v:OutputLayer => Iterable[Factor]()
  }
}

object SentimentRNN extends FastLogging {
  def train(train:File,test:File,dev:File) {
    NNUtils.setTensorImplementation(NNUtils.JBLAS)

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
        new model.BackPropagationExample(in,out)//,Random.nextDouble() >= 0.99) //gradient check at every 100th example, expensive
      }.toList
    }
    val trainExamples = createExamples(trainTrees)
    val testExamples = testTrees.map(t => model.createNetwork(t))
    val devExamples = devTrees.map(t => model.createNetwork(t)).toSeq
    val devTotalFine = devExamples.map(_._2.size).sum.toDouble
    val devTotal = devExamples.size.toDouble

    //train the model
    //Gradient check
    //Example.testGradient(model.parameters,trainExamples.head)
    var iterations = 0
    while(iterations < 100) {
      iterations += 1
      val trainer = new ParallelBatchTrainer(model.parameters, maxIterations = -1, optimizer = new AdaGrad(0.01))
      trainExamples.grouped(27).foreach(e => trainer.processExamples(e))
      val objectiveDevSet = devExamples.map(e => model.totalObjective(e._1,e._2)).sum
      val fineGrained = devExamples.map(_._2.count(e => e.target.value.maxIndex == e.value.maxIndex)).sum / devTotalFine
      val total = devExamples.count(t => {
        val l = t._2.find(_.factor._1.parentFactor == null).get
        l.target.value.maxIndex == l.value.maxIndex
      }) / devTotal
      logger.info(
        s"""Objective on dev set: $objectiveDevSet
           |Finegrained on dev set: $fineGrained
           |Accuracy on dev set: $total
         """.stripMargin)
    }
    val objectiveTestSet = testExamples.map(e => model.totalObjective(e._1,e._2)).sum
    logger.info(s"Objective on test set: $objectiveTestSet")
  }

  def main (args: Array[String]) {
    val Array(training,test,dev) = args
    train(new File(training),new File(test),new File(dev))
  }
}
