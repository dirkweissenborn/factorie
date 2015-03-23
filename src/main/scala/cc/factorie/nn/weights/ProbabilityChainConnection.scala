package cc.factorie.nn.weights

import cc.factorie.la._
import cc.factorie.maths
import cc.factorie.model._
import cc.factorie.nn._

//Copied parts mainly from ChainModel; most of the computation here is done in log space
trait ProbabilityChainConnection[N1<:NNUnit] extends NNConnection {
  override type ConnectionType = ProbabilityChainConnection[N1]
  type LayerType = Layer
  override def weights: Weights2
  override val numUnits: Int = -1
  override final val numOutputUnits: Int = Int.MaxValue
  override final protected[nn] def _outputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = Seq.empty[NNUnit]
  override final protected[nn] def _inputUnits(c: ConnectionType#LayerType): Seq[NNUnit] = c.asInstanceOf[Layer]._1

  case class Layer(_1:Seq[N1]) extends super.Layer {
    override def numUnits: Int = _1.size
  }
  override def newConnection(units: Seq[NNUnit]): ConnectionType#LayerType = {
    assert(units.forall(_.activationFunction == ActivationFunction.Linear),
      "Activation of units in a linear chain should be linear, since activation is handled by the chain!")
    val c = Layer(units.asInstanceOf[Seq[N1]])
    c.setFeature("alphas",Array.fill(units.size)(
      TensorUtils.fillDense(units.head.value.dim1)(_ =>Double.NegativeInfinity)))
    c.setFeature("betas",Array.fill(units.size)(
      TensorUtils.fillDense(units.head.value.dim1)(_ =>Double.NegativeInfinity)))
    c
  }

  abstract override def reset(c: ConnectionType#LayerType): Unit = {
    val alphas = c.getFeature[Array[Tensor1]]("alphas").get
    val betas = c.getFeature[Array[Tensor1]]("betas").get
    alphas.foreach(a => a.foreachActiveElement((i,_) => a(i) = Double.NegativeInfinity))
    betas.foreach(a => a.foreachActiveElement((i,_) => a(i) = Double.NegativeInfinity))
    super.reset(c)
  }
  override protected[nn] def _forwardPropagate(c: ConnectionType#LayerType): Unit = {
    val units = c._1
    if (units.length == 0) return
    val alphas = c.getFeature[Array[Tensor1]]("alphas").get
    val betas = c.getFeature[Array[Tensor1]]("betas").get
    val logZ = inferFast(units,alphas,betas)
    c.setFeature("logZ",logZ)
    val len = units.length
    var i = 0
    while (i < len) {
      val u = units(i)
      val curAlpha = alphas(i)
      val curBeta = betas(i)
      u.value.zero()
      u.value += curAlpha
      u.value += curBeta
      u match {
        case u:LabeledNNUnit => u.setLastObjective((u.target.value dot u.value)-logZ)
        case _ => //
      }
      u.value.expNormalize(logZ)
      i += 1
    }
  }

  override protected[nn] def _backPropagate(c: ConnectionType#LayerType): Unit = {}
  
  override protected[nn] def _backPropagateGradient(c: ConnectionType#LayerType): Tensor = {
    val units = c._1.asInstanceOf[Seq[LabeledNNUnit]]
    val len = units.length
    var i = 0
    val transGradient = c.tempWeightGradient.asInstanceOf[Tensor2]
    val gs = c.tempGradients.toSeq
    val alphas = c.getFeature[Array[Tensor1]]("alphas").get
    val betas = c.getFeature[Array[Tensor1]]("betas").get
    val logZ = c.getFeature[Double]("logZ").get
    while (i < len) {
      val u = units(i)
      val g = gs(i)
      val prevAlpha = if (i >= 1) alphas(i - 1) else null.asInstanceOf[Tensor1]
      val prevU = if (i >= 1) units(i - 1) else null.asInstanceOf[LabeledNNUnit]
      val curBeta = betas(i)
      val curLocalScores = u.input
      
      g -= u.value
      g += u.target.value
      u.incrementObjectiveGradient(g)
      
      if (i >= 1) {
        var ii = 0
        while (ii < u.value.dim1) {
          var jj = 0
          while (jj < u.value.dim1) {
            transGradient(ii, jj) += 
              -math.exp(prevAlpha(ii) + weights.value(ii, jj) + curBeta(jj) + curLocalScores(jj) - logZ) + 
                u.target.value(jj) * prevU.target.value(ii)
            jj += 1
          }
          ii += 1
        }
      }
      i += 1
    }
    transGradient
  }
  
  def inferFast(units: Seq[NNUnit], alphas:Array[Tensor1], betas:Array[Tensor1]): Double = {
    val d1 = weights.value.dim1
    alphas(0) := units.head.input
    var i = 1
    val tmpArray = Array.fill(d1)(0.0)
    while (i < units.size) {
      val ai = alphas(i)
      val aim1 = alphas(i - 1)
      var vi = 0
      while (vi < d1) {
        var vj = 0
        while (vj < d1) {
          tmpArray(vj) = weights.value(vj,vi) + aim1(vj)
          vj += 1
        }
        ai(vi) = maths.sumLogProbs(tmpArray)
        vi += 1
      }
      alphas(i) += units(i).input
      i += 1
    }
    betas.last.zero()
    i = units.size - 2
    while (i >= 0) {
      val bi = betas(i)
      val bip1 = betas(i + 1)
      val lsp1 = units(i + 1).input
      var vi = 0
      while (vi < d1) {
        var vj = 0
        while (vj < d1) {
          tmpArray(vj) = weights.value(vi, vj) + bip1(vj) + lsp1(vj)
          vj += 1
        }
        bi(vi) = maths.sumLogProbs(tmpArray)
        vi += 1
      }
      i -= 1
    }
    maths.sumLogProbs(alphas.last.asArray)
  }
}


