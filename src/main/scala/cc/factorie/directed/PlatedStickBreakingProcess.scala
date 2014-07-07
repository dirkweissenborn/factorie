package cc.factorie.directed

import cc.factorie.variable._
import scala.util.Random

/**
 * Created by Dirk Weissenborn on 30.06.14.
 */
// generate a discrete value from masses and, pi and m as parameters of this process
object PlatedStickBreakingProcess extends DirectedFamily4[SeqVar[IntegerVar],MassesVar,RealVar,RealVar] {
  override def newFactor(c: PlatedStickBreakingProcess.C, p1: PlatedStickBreakingProcess.P1, p2: PlatedStickBreakingProcess.P2, p3: PlatedStickBreakingProcess.P3): Factor = new Factor(c,p1,p2,p3)

  class Factor(override val _1:SeqVar[IntegerVar],override val _2:MassesVar,override val _3:RealVar,override val _4:RealVar) extends super.Factor(_1,_2,_3,_4) with SeqGeneratingFactor {
    def counts = _2
    def m = _3
    def pi = _4
    def countsFrom(k:Int) = {
      var i = k
      var sum = 0.0
      while(i<counts.value.length) {
        sum+=counts.value(i)
        i+=1
      }
      sum
    }
    override def proportionalForChildIndex(idx: Int): Double = {
      var level = _1(idx).intValue
      var fromCount = countsFrom(level)
      var product = (m.doubleValue*pi.doubleValue+counts(level))/(pi.doubleValue+fromCount)
      val _mPi = (1.0-m.doubleValue)*pi.doubleValue
      level -= 1
      while(level>=0) {
        product *= math.pow(_mPi + fromCount,done)
        fromCount += counts(i)
        product /= math.pow(pi.doubleValue + fromCount,done)
        level-= 1
      }
      product
    }
    def probabilitiesForLevels(maxLevel:Int) = {
      val result = new Array[Double](maxLevel + 1)
      var level = 0
      var fromCount = countsFrom(level)
      var product = 1.0
      val _mPi = (1.0-m.doubleValue)*pi.doubleValue
      while(level <= maxLevel) {
        result(level) = product * (m.doubleValue*pi.doubleValue+counts(level))/(pi.doubleValue+fromCount)
        product /= (pi.doubleValue + fromCount)
        fromCount -= counts(i)
        product *= (_mPi + fromCount)
        level+= 1
      }
      product
    }
    //efficiently sampling levels associated with this process jointly
    override def sampledValue(counts: MassesVar#Value, m: RealVar#Value, pi: RealVar#Value)(implicit random: Random): SeqVar[IntegerVar]#Value = {
      val result = new SeqVariable[MutableIntegerVar](Array.tabulate(_1.length)(_ => new IntegerVariable()))
      val rs = Array.tabulate(_1.length)(_ => random.nextDouble())
      var k = 0
      var fromCount = countsFrom(k)
      while(rs.exists(_ >= 0.0)) {
        val v = (m.doubleValue*pi.doubleValue+counts(k))/(pi.doubleValue+fromCount)
        (0 until rs.length).foreach(i => {
          if (v > rs(i)) {
            rs(i) -= v
            result(i).set(k)(null)
          }
          else {
            rs(i) *= (pi.doubleValue + fromCount)
            fromCount -= counts(k)
            rs(i) /= ((1.0 - m.doubleValue) * pi.doubleValue + fromCount)
          }
        })
        k+=1
      }
      rs
    }
    //efficiently calculating probability for levels associated with this process jointly
    override def pr(children: SeqVar[IntegerVar]#Value, counts: MassesVar#Value, m: RealVar#Value, pi: RealVar#Value): Double = {
      var sortedLevels = children.map(_.intValue).sortBy(-_)
      var product = 1.0
      val _mPi = (1.0-m.doubleValue)*pi.doubleValue
      var i = sortedLevels.head
      var fromCount = countsFrom(i+1)
      var levelsToProcess = sortedLevels.length
      var done = 0.0
      while(i>=0) {
        sortedLevels = sortedLevels.dropWhile(_ == i)
        var doneNow = levelsToProcess - sortedLevels.length
        if(doneNow > 0)
          product *= math.pow((m.doubleValue*pi.doubleValue+counts(i))/(pi.doubleValue+fromCount),doneNow)
        if(done > 0)
          product *= math.pow(_mPi + fromCount,done)
        fromCount += counts(i)
        if(done > 0)
          product /= math.pow(pi.doubleValue + fromCount,done)
        i-= 1
        levelsToProcess = sortedLevels.length
        done += doneNow
      }
      product
    }
    override def updateCollapsedParentsForIdx(weight: Double, idx: Int): Boolean = counts.value += (child(idx).intValue,weight)
    override def updateCollapsedParents(weight: Double): Boolean = { child.foreach(iVar => counts.value += (iVar.intValue,weight));true}
  }

}
