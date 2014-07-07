package cc.factorie.directed

import cc.factorie.variable._
import scala.util.Random
import scala.annotation.tailrec
import cc.factorie.la.GrowableDenseTensor1

/**
 * Created by Dirk Weissenborn on 24.06.14.
 */
/**
 * Consists of a level (DiscreteVar) and a path (TreePathVar) the level refers to, which together induce a probability
 * distribution (taken from a Mixture[ProportionsVariable]) over features (child: DiscreteVar).
 */
object PlatedNestedCRP extends DirectedFamily4[DiscreteSeqVar,Mixture[ProportionsVariable],SeqVar[IntegerVar],TreePathVar] {
  override def newFactor(c: DiscreteVar, p1: Mixture[ProportionsVariable], p2:SeqVar[IntegerVar] , p3: TreePathVar): PlatedNestedCRP.Factor =
    new Factor(c,p1,p2,p3)
  
  class Factor(_1:DiscreteSeqVar,_2:Mixture[ProportionsVariable],_3:SeqVar[IntegerVar],_4:TreePathVar) extends super.Factor(_1,_2,_3,_4) with SeqGeneratingFactor with SeqParentFactor {
    def mixture = _2
    def path = _4
    def levels = _3

    override def sampledValue(p1: Mixture[ProportionsVariable]#Value, p2:SeqVar[IntegerVar]#Value , p3: TreePathVar#Value)(implicit random: Random): DiscreteSeqVar#Value =
      p2.map(level => _1.domain.elementDomain(p1(p3.valueAtLevel(level.intValue)).sampleIndex)).asInstanceOf[DiscreteSeqVar#Value]

    override def pr(v1: DiscreteSeqVar#Value, v2: Mixture[ProportionsVariable]#Value, v3:SeqVar[IntegerVar]#Value, v4: TreePathVar#Value): Double =
      v1.foldLeft(1.0)((acc,v) => acc * v2(v4.valueAtLevel(v3)).apply(v.intValue))

    override def proportional: Double = _1.zip(levels).foldLeft(1.0)((acc,v) => acc * mixture(path.valueAtLevel(v._2.value)).value.masses(v._1.intValue))

    /** Update sufficient statistics in collapsed parents, using current value of child, with weight.  Return false on failure. */
    override def updateCollapsedParents(weight: Double): Boolean = {
      _1.zip(levels).foreach{ case (child,level)=> mixture(path.value.valueAtLevel(level.intValue)).incrementMasses(child.intValue,weight)(null) }
      true
    }

    //Use the actual probability here, because it is needed during collapsed gibbs sampling
    override def proportionalForChildIndex(idx: Int): Double = mixture(path.valueAtLevel(levels(idx).value)).value(_1(idx).intValue)
    override def proportionalForParentIndex(idx: Int): Double = proportionalForChildIndex(idx)
    override def updateCollapsedParentsForIdx(weight: Double, idx: Int): Boolean = mixture(path.value.valueAtLevel(levels(idx).intValue)).incrementMasses(child(idx).intValue,weight)(null)
  }
}

object NestedCRPPrior extends DirectedFamily2[TreePathVar,NestedCRPCountsVariable] {
  class Factor(_1:TreePathVar,_2:NestedCRPCountsVariable) extends super.Factor(_1,_2) with DiscreteGeneratingFactor {
    override def prValue(value: Int): Double = _2.value.pr(value)

    override def sampledValue(p1: NestedCRPCounts)(implicit random: Random): TreePath =
      throw new NotImplementedError("Not implemented yet; might be difficult to implement without level of path because trees are infinite")

    override def pr(v1: TreePath, v2: NestedCRPCounts): Double = v2.pr(v1.intValue)

    /** Update sufficient statistics in collapsed parents, using current value of child, with weight.  Return false on failure. */
    override def updateCollapsedParents(weight: Double): Boolean = {
      var c = _1.value
      while(c != null) {
        _2.value += (c.intValue,weight)
        c = c.predecessor
      }
      true
    }
  }

  override def newFactor(c: NestedCRPPrior.C, p1: NestedCRPPrior.P1): Factor = new Factor(c,p1)
}

//At every node in the tree we have a CRP; the probability for a specific path is the conditional probability of the end node of this path given the path before
class NestedCRPCounts(val gamma:Double, pathDomain:TreePathDomain) extends GrowableDenseMasses1(pathDomain) {
  final override def pr(i: Int): Double =
    if(i == pathDomain.root.intValue)
      massTotal/(massTotal+gamma)
    else {
      val c = pathDomain(i)
      val pre = c.predecessor.intValue
      apply(i)/(apply(pre)+gamma)*pr(pre)
    }
  //use this for sampling, because when using pr everything gets calculated many times
  def prs(maxDepth:Int) = (1 until pathDomain.size).foldLeft(Map[Int,Double](0->(massTotal/(massTotal+gamma))))((acc,i) => {
    val c = pathDomain(i)
    val pre = c.predecessor.intValue
    if(c.depth <= maxDepth) acc + (i -> apply(i)/(apply(pre)+gamma)*acc.getOrElse(pre,pr(pre))) else acc
  })
}
class NestedCRPCountsVariable(counts:NestedCRPCounts) extends VarWithDeterministicValue {
  override type Value = NestedCRPCounts
  /** Abstract method to return the value of this variable. */
  override def value: Value = counts
}

trait TreePathVar extends MutableDiscreteVar {
  override def domain: TreePathDomain
  override type Value = TreePath
}
//A path only needs to know about its end and predecessor (much like a reversed list)
trait TreePath extends DiscreteValue {
  self =>
  override def stringPrefix: String = "TreePath"
  def predecessor:TreePath
  lazy val depth:Int = if(predecessor == null) 0 else predecessor.depth + 1
  def valueAtLevel(level:Int) = {
    if(level > depth) -1
    else {
      var c = this
      while (c.depth > level)
        c = c.predecessor
      c.intValue
    }
  }
  def path:List[Int] = {
    var list = List(singleIndex)
    var c = predecessor
    while(c != null) {
      list ::= c.singleIndex
      c = c.predecessor
    }
    list
  }
  def iterator = new Iterator[TreePath]{
    var c = self
    override def hasNext: Boolean = c != null
    override def next(): TreePath = {
      val next = c
      c = c.predecessor
      next
    }
  }
}

class TreePathDomain extends DiscreteDomain(-1) {
  thisDomain =>
  override type Value = TreePath
  //create root
  val root = newTreePath(null)

  override def length: Int = _elements.size

  def newTreePath(predecessor:TreePath = root) = {
    val newPath = new DiscreteValue(this.size,predecessor)
    _elements += new DiscreteValue(this.size,predecessor)
    newPath
  }

  protected class DiscreteValue(singleIndex:Int, _predecessor: TreePath = root) extends super.DiscreteValue(singleIndex) with TreePath {
    override def predecessor: TreePath = _predecessor
  }
}