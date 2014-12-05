package cc.factorie.nn

import cc.factorie.la.DenseTensor1
import cc.factorie.nn.sentiment.{LoadPTB, SentimentRNN}
import cc.factorie.optimize.OptimizableObjectives
import cc.factorie.variable.CategoricalDomain
import org.junit.runner.RunWith
import org.scalatest._
import org.scalatest.junit.JUnitRunner

import scala.util.Random

/**
 * Created by diwe01 on 24.07.14.
 */
@RunWith(classOf[JUnitRunner])
class TestFeedForwardNeuralNetwork extends FlatSpec {
  self =>

  val tokenDomain = new CategoricalDomain[String]()
  tokenDomain.index("a")
  tokenDomain.index("b")
  tokenDomain.index("c")
  tokenDomain.freeze()
//(2 (4 a)
  val pt = LoadPTB.sentimentPTPFromString("(2 (4 a) (1 (0 b) (2 c)))")
 // NNUtils.setTensorImplementation(NNUtils.JBLAS)

  "Gradient checks" should "not fail for a basic feed forward neural network" in {
    val model = new BasicFeedForwardNN(Array((2,ActivationFunction.Tanh),(10,ActivationFunction.Sigmoid),(1,ActivationFunction.Sigmoid)))
    val nets = Seq(
      model.createNetwork(NNUtils.fillDense(2)(_ => Random.nextDouble()-0.5),NNUtils.fillDense(1)(_ => 1.0)),
      model.createNetwork(NNUtils.fillDense(2)(_ => Random.nextDouble()-0.5),NNUtils.fillDense(1)(_ => -1.0)),
      model.createNetwork(NNUtils.fillDense(2)(_ => Random.nextDouble()-0.5),NNUtils.fillDense(1)(_ => -1.0)),
      model.createNetwork(NNUtils.fillDense(2)(_ => Random.nextDouble()-0.5),NNUtils.fillDense(1)(_ => 1.0)))

    val examples = nets.map(net => new model.BackPropagationExample(net))
    assert(examples.forall(_.checkGradient()))
  }
  it should "not fail on a simple sentiment RNN either" in {
    val sentiModel = new SentimentRNN(10,tokenDomain,withTensors = false)
    sentiModel.objectiveF = OptimizableObjectives.squaredMultivariate
    val net = sentiModel.createNetwork(pt)
    val example = new sentiModel.BackPropagationExample(net)
    assert(example.checkGradient())
  }
  it should "not fail on a sentiment tensor RNN either" in {
    val sentiModel2 = new SentimentRNN(2, tokenDomain)
    sentiModel2.objectiveF = OptimizableObjectives.squaredMultivariate
    val net = sentiModel2.createNetwork(pt)
    val example = new sentiModel2.BackPropagationExample(net)
    assert(example.checkGradient())

    /*NNUtils.setTensorImplementation(NNUtils.JBLAS)
    val sentiModel3 = new SentimentRNN(2,tokenDomain)
    val (in3,out3) = sentiModel3.createNetwork(pt)
    sentiModel3.forwardAndBackPropagateOutputGradient(in3)
    sentiModel3.bias.weights.value := sentiModel2.bias.weights.value
    sentiModel3.childWeights1.weights.value := sentiModel2.childWeights1.weights.value
    sentiModel3.childWeights2.weights.value := sentiModel2.childWeights2.weights.value
    sentiModel3.tensorWeights.weights.value := sentiModel2.tensorWeights.weights.value
    sentiModel3.outBias.weights.value := sentiModel2.outBias.weights.value
    sentiModel3.embeddings.weights.value := sentiModel2.embeddings.weights.value
    sentiModel3.outputWeights.weights.value := sentiModel2.outputWeights.weights.value

    val example3 = new sentiModel3.BackPropagationExample(in3,out3)
    val g1 = sentiModel2.forwardAndBackPropagateOutputGradient(in)
    val g2 = sentiModel3.forwardAndBackPropagateOutputGradient(in3)
    assert(example3.checkGradient())*/
  }
  it should "not fail on a simple sentiment RNN with JBLAS either" in {
    NNUtils.setTensorImplementation(NNUtils.JBLAS)
    val sentiModel = new SentimentRNN(10,tokenDomain,withTensors = false)
    sentiModel.objectiveF = OptimizableObjectives.squaredMultivariate
    val network = sentiModel.createNetwork(pt)
    val example = new sentiModel.BackPropagationExample(network)
    //assert(example.checkGradient())
  }
  it should "not fail on a sentiment tensor RNN with JBLAS either" in {
    NNUtils.setTensorImplementation(NNUtils.JBLAS)
    val sentiModel = new SentimentRNN(2,tokenDomain)
    sentiModel.objectiveF = OptimizableObjectives.squaredMultivariate
    val network = sentiModel.createNetwork(pt)
    val example = new sentiModel.BackPropagationExample(network)
    assert(example.checkGradient())
  }

  /*it should "not fail on a tensor sentiment RNN either" {

  }*/
}
