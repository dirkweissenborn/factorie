/* Copyright (C) 2008-2014 University of Massachusetts Amherst.
   This file is part of "FACTORIE" (Factor graphs, Imperative, Extensible)
   http://factorie.cs.umass.edu, http://github.com/factorie
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

package cc.factorie.directed

import cc.factorie.directed.PlatedNestedCRP
import cc.factorie.infer._
import scala.collection.mutable.{ArrayBuffer, HashSet}
import cc.factorie.variable._
import scala.collection.mutable
import cc.factorie.util.FastLogging
import scala.Some
import cc.factorie.la.Tensor
import scala.util.Random

/**
 * This implementation is based on a sequence of handlers, much like a chain of responsibility, from which the first one
 * that can handle a variable embedded in a model is chosen to sample the variable in question.
 * The sequences of handlers can be extended with self implemented handlers, to handle special unhandled cases or cases that
 * could be more efficient than by default (which usually samples from the product of all factor proportionals of a variable).
 *
 * @param collapse Predefined set of collapsed variables that are used globally in this model (usually some mixtures over proportions),
 *                 Factors have to know how to update its collapsed variables.
 * @param model
 * @param samplingCandidates A function that assigns candidates for sampling to a variable (Optional).
 *                           If it returns null for a variable, the whole domain is used as candidate set (default, can be slow).
 *                           For SeqVars this function should return candidates for one element of the SeqVar!
 *                           Candidates should inherit from Tensor.
 *                           Use for example DiscreteValue for discrete variable candidates, or RealValue for real variable candidates.
 * @param random
 */
//TODO memory efficient implementation for incremental/online sampling. At the moment every collapsed variable has to be in memory before creating this sampler. - dirk
//HACKY solution: This can be circumvented manually by collapsing new collapsed variables (e.g. theta of new LDA document) and putting them into the collapsed set, and later removing them again to stay memory efficient.
class CollapsedGibbsSampler(collapse: Iterable[Var], val model: DirectedModel, samplingCandidates: MutableVar => Seq[Tensor] = _ => null)(implicit val random: scala.util.Random)
  extends Sampler[Iterable[MutableVar]] {
  var debug = false
  makeNewDiffList = false // override default in cc.factorie.Sampler
  var temperature = 1.0 // TODO Currently ignored?
  def defaultHandlers = ArrayBuffer[CollapsedGibbsSamplerHandler](
    DefaultDiscreteGibbsSamplerHandler
  )
  private val handlers = defaultHandlers
  def appendHandler(h:CollapsedGibbsSamplerHandler) = handlers.append(h)
  def preprendHandler(h:CollapsedGibbsSamplerHandler) = handlers.prepend(h)

  val cacheClosures = true
  val closures = new mutable.HashMap[Var, CollapsedGibbsSamplerClosure]
  val collapsed = new HashSet[Var] ++ collapse

  // Initialize collapsed parameters specified in constructor
  val collapser = new Collapse(model)
  collapse.foreach(v => collapser(Seq(v)))

  def isCollapsed(v:Var): Boolean = collapsed.contains(v)

  def process1(v:Iterable[MutableVar]): DiffList = {
    val d = newDiffList
    // If we have a cached closure, just use it and return
    if (cacheClosures && v.size == 1 && closures.contains(v.head)) {
      closures(v.head).sample(d)
    } else {
      // Get factors, no guarantees about their order
      var done = false
      val handlerIterator = handlers.iterator
      while (!done && handlerIterator.hasNext) {
        val closure = handlerIterator.next().sampler(v, this, samplingCandidates)
        if (closure ne null) {
          done = true
          closure.sample(d)
          if (cacheClosures && v.size == 1) {
            closures(v.head) = closure
          }
        }
      }
      if (!done) throw new Error("CollapsedGibbsSampler: No sampling method found for variable "+v+" with factors "+model.factors(v).map(_.getClass.getName).mkString(", "))
    }
    d
  }

  /** Convenience for sampling single variable */
  def process(v:MutableVar): DiffList = process(Seq(v))
}

trait CollapsedGibbsSamplerHandler {
  /**
   * @param vs variables to sample, usually one, but for blocked sampling there could potentially be more
   * @param sampler contains model, collapsed variables
   * @param samplingCandidates candidates from which to sample, if null is returned, entire domain will be considered
   * @param random
   * @return a closure that encapsulates basically just a sampling function, with potential configuration
   */
  def sampler(vs:Iterable[MutableVar], sampler:CollapsedGibbsSampler, samplingCandidates: MutableVar => Seq[Tensor])(implicit random: scala.util.Random): CollapsedGibbsSamplerClosure
}
trait CollapsedGibbsSamplerClosure {
  def sample(implicit d:DiffList = null): Unit
}

//proportionals of candidates for sampling are products of all neighbouring factor proportionals
//Can handle all basic discrete variables, though some might not be very efficient
//TODO find a way to extend this handler to support efficient sampling with MultinomialFromSeq.Factor
object DefaultDiscreteGibbsSamplerHandler extends CollapsedGibbsSamplerHandler {
  def sampler(vs:Iterable[MutableVar], sampler:CollapsedGibbsSampler, samplingCandidates:MutableVar => Seq[Tensor])(implicit random: scala.util.Random) = {
    try {
      if(vs.size != 1) null
      else
        vs.head match {
          case v:MutableDiscreteVar => new DiscreteClosure(v,sampler,samplingCandidates(v).asInstanceOf[Seq[DiscreteValue]])
          case v:MutableDiscreteSeqVar[_] => new DiscreteSeqClosure(v,sampler,samplingCandidates(v).asInstanceOf[Seq[DiscreteValue]])
          case _ => null
      }
    } catch {
      case e:java.lang.ClassCastException => throw new ClassCastException("Sampling candidates of DiscreteVar have to be of type DiscreteValue")
    }
  }

  class DiscreteClosure(v:MutableDiscreteVar,sampler:CollapsedGibbsSampler, samplingCandidates: Seq[DiscreteValue])(implicit random: scala.util.Random) extends CollapsedGibbsSamplerClosure {
    val model = sampler.model
    val childFactors = model.childFactors(v)
    val parentFactor = model.getParentFactor(v)
    val collapsedFactors = ArrayBuffer[DirectedFactor]()
    collapsedFactors ++= childFactors.filter(f => f.parents.exists(sampler.isCollapsed))
    if (parentFactor.isDefined && parentFactor.get.parents.exists(sampler.isCollapsed))
      collapsedFactors += parentFactor.get

    val candidates =
      if (samplingCandidates == null) 0 until v.domain.size
      else samplingCandidates.map(_.intValue)

    //reusable
    val distribution = Array.ofDim[Double](candidates.size)

    def sample(implicit d:DiffList = null) {
      collapsedFactors.foreach(_.updateCollapsedParents(-1.0))
      var sum = 0.0
      var i = 0
      while(i < distribution.length) {
        val candidate = candidates(i)
        v.set(candidate)(null)
        val pValue = parentFactor.fold(1.0)(_.proportional)
        val cValue = childFactors.foldLeft(1.0)(_ * _.proportional)
        val pr = pValue * cValue
        sum += pr
        distribution(i) = pr
        i+=1
      }
      if (sum == 0) v.set(candidates(random.nextInt(distribution.length)))(null)
      else v.set(candidates(cc.factorie.maths.nextDiscrete(distribution, sum)(random)))(null)
      collapsedFactors.foreach(_.updateCollapsedParents(1.0))
    }
  }
  class DiscreteSeqClosure(v:MutableDiscreteSeqVar[_],sampler:CollapsedGibbsSampler, samplingCandidates: Seq[DiscreteValue])(implicit random: scala.util.Random) extends CollapsedGibbsSamplerClosure with FastLogging {
    val model = sampler.model
    val childFactors = model.childFactors(v)
    val parentFactor = model.getParentFactor(v)
    val candidates =
      if (samplingCandidates == null) 0 until v.domain.elementDomain.size
      else samplingCandidates.map(_.intValue)
    val collapsedFactors = ArrayBuffer[DirectedFactor]()
    collapsedFactors ++= childFactors.filter(f => f.parents.exists(sampler.isCollapsed))
    if (parentFactor.isDefined && parentFactor.get.parents.exists(sampler.isCollapsed))
      collapsedFactors += parentFactor.get
    //Reusable
    val distribution = Array.ofDim[Double](candidates.length)

    def sample(implicit d:DiffList = null) {
      (0 until v.size).foreach(idx => {
        var sum = 0.0
        //update sufficient statistics
        updateCollapsedParents(-1.0,idx)
        (0 until distribution.length).foreach {
          i =>
            val candidate = candidates(i)
            v.set(idx, candidate)(null)
            val pValue = parentFactor match {
              //Fast because it only calculates probability for current index
              case Some(f: SeqGeneratingFactor) => f.proportionalForChildIndex(idx)
              //Defaults that could potentially be very slow if we are sampling a SeqVar (e.g., above two cases)
              case Some(f: DirectedFactor) =>
                logger.warn(s"Sampling DiscreteSeqVar using ${f.getClass} could be slow. ${f.getClass} should inherit from DiscreteSeqGeneratingFactor!")
                f.pr
              case None => 1.0
            }
            val cValue = childFactors.foldLeft(1.0) {
              //useful for PlatedDiscreteMixture
              case (acc: Double, f: SeqParentFactor) => acc * f.proportionalForParentIndex(idx)
              //Defaults that could potentially be very slow if we are sampling a SeqVar
              case (acc: Double, f: DirectedFactor) => acc * f.pr
            }
            val pr = pValue * cValue
            sum += pr
            distribution(i) = pr
        }
        val selected =
          if (sum == 0)  candidates(random.nextInt(candidates.length))
          else candidates(cc.factorie.maths.nextDiscrete(distribution, sum)(random))
        v.set(idx,selected)(null)
        //update sufficient statistics
        updateCollapsedParents(1.0,idx)
      })
    }
    def updateCollapsedParents(weight:Double, idx:Int) {
      collapsedFactors.foreach {
        case f: SeqGeneratingFactor if parentFactor.exists(_ == f) => f.updateCollapsedParentsForIdx(weight, idx)
        case f: SeqParentFactor => f.updateCollapsedParentsForParentIdx(weight, idx)
        case f: DirectedFactor => throw new Error("Factors connected to discrete sequence variables should implement either SeqGeneratingFactor (for parent factors of such) or SeqParentFactor (for child factors), to update collapsed parents efficiently!")
      }
    }
  }
}

/**
 * This handler is specifically designed for working with nested chinese restaurant processes. It efficiently samples a new path, given
 * the current level assignments, only considering paths to the maximum level of all level variables. If an inner node is sampled, a new path
 * to the maximum depth starting at the sampled inner node is created.
 * Following http://cs.brown.edu/courses/csci2950-p/fall2011/lectures/2011-10-25_bryantWang.pdf & http://www.cs.cmu.edu/~cchua/papers/amazon_appendix.pdf
 */
object NestedCRPCollapsedGibbsSamplerHandler extends CollapsedGibbsSamplerHandler {
  override def sampler(vs: Iterable[MutableVar], sampler: CollapsedGibbsSampler, samplingCandidates: (MutableVar) => Seq[Tensor])(implicit random: Random) = {
    try {
      if(vs.size != 1 || !sampler.model.childFactors(vs.head).exists(_.isInstanceOf[PlatedNestedCRP.Factor])) null
      else
        vs.head match {
          case v:TreePathVar => new Closure(v,sampler,samplingCandidates(v).asInstanceOf[Seq[TreePath]])
          case v:MutableSeqVar[IntegerVar] => new LevelClosure(v,sampler,samplingCandidates(v).asInstanceOf[Seq[Int]])
          case _ => null
        }
    } catch {
      case e:java.lang.ClassCastException => throw new ClassCastException("Sampling candidates of DiscreteVar have to be of type DiscreteValue")
    }
  }

  class Closure(v:TreePathVar,sampler:CollapsedGibbsSampler, samplingCandidates: Seq[DiscreteValue])(implicit random: scala.util.Random) extends CollapsedGibbsSamplerClosure {
    val model = sampler.model
    val (allCrpFactors, childFactors) = model.childFactors(v).partition(_.isInstanceOf[PlatedNestedCRP.Factor])
    val parentFactor = model.getParentFactor(v)
    val collapsedFactors = ArrayBuffer[DirectedFactor]()
    collapsedFactors ++= childFactors.filter(f => f.parents.exists(sampler.isCollapsed))
    if (parentFactor.isDefined && parentFactor.get.parents.exists(sampler.isCollapsed))
      collapsedFactors += parentFactor.get
    val candidates =
      if (samplingCandidates == null) 0 until v.domain.size
      else samplingCandidates.map(_.intValue)
    //This is needed if this TreePathVar is part of different factor families in the model (one family <-> one mixture variable)
    val nCrpFactorsByMixture = allCrpFactors.groupBy(_.asInstanceOf[PlatedNestedCRP.Factor].mixture)
    //reusable
    val distribution = Array.ofDim[Double](candidates.size)

    def sample(implicit d: DiffList = null) {
      collapsedFactors.foreach(_.updateCollapsedParents(-1.0))
      var sum = 0.0
      // calculate level to feature counts which are needed for sampling, for each nCRP family in the model
      // level -> features -> counts; feature=-1 contains total local count of level
      val levelFeatureCounts = nCrpFactorsByMixture.mapValues(_.foldLeft(mutable.Map[Int, mutable.Map[Int, Int]]())((counts,factor) => {
        val f = factor.asInstanceOf[PlatedNestedCRP.Factor]
        val feature = f.child.intValue
        val levels = f.levels
        levels.foreach(level => {
          val levelCounts = counts.getOrElseUpdate(level.intValue, mutable.Map[Int, Int](-1 -> 0))
          levelCounts(-1) += 1
          levelCounts += feature -> (1 + levelCounts.getOrElse(feature, 0))
        })
        counts
      }))
      val maxLevel = allCrpFactors.foldLeft(0)((max,f)=> math.max(f.asInstanceOf[PlatedNestedCRP.Factor].levels.value.maxBy(_.value).value,max))
      //pre-calculate path priors if possible, because their calculation is recursive it is better to pre-calculate them
      val getParentScore:()=>Double = parentFactor match {
        case Some(nCrpPrior:NestedCRPPrior.Factor) =>
          val nCRPCounts = nCrpPrior._2.asInstanceOf[NestedCRPCountsVariable].value
          val probs = nCRPCounts.prs(maxLevel)
          () => v.value.depth match {
            case d:Int if d == maxLevel => probs(v.intValue)
            case _ => probs(v.intValue) * nCRPCounts.gamma / (nCRPCounts(v.intValue)+nCRPCounts.gamma) // because we need to create a new path of length maxLength if this partial path gets sampled
          }
        case _ => () => parentFactor.fold(1.0)(_.proportional)
      }
      var i = 0
      while(i < distribution.length) {
        val candidate = candidates(i)
        v.set(candidate)(null)
        if(v.value.depth > maxLevel) distribution(i) = 0.0
        else {
          val pValue = getParentScore()
          val cValue = childFactors.foldLeft(1.0)(_ * _.proportional)
          val nCRPValue = nCrpFactorsByMixture.foldLeft(1.0) { case (totalProduct, (mixture, nCrpFactors)) =>
            //needed for sampling potential new paths
            val uniformProportions = new MassesProportions1({mixture.head.value match {
              case withPrior: DirichletPrior if withPrior.prior != null => withPrior.prior.asInstanceOf[Masses1]
              case m: Proportions1 => new UniformMasses1(m.dim1, 0.0)
              case _ => throw new IllegalArgumentException("Mixture over 1-dimensional proportions needed for CRP Factors")//will never be the case
            }})
            //Basically for each feature-assignment pair we need to calculate gamma of global+local divided by gamma of global count (+ prior)
            //=> (global+prior)*(global+prior+1)*...*(global+prior+local-1)
            totalProduct * levelFeatureCounts(mixture).foldLeft(1.0) {
              case (product, (level, counts)) =>
                val assignment = v.value.valueAtLevel(level) //gives -1 if level is below this paths level
                var i = 0
                val m = if (assignment >= 0) mixture(assignment).value else uniformProportions //sample from uniformProps if this node in the tree is new
                product * counts.foldLeft(1.0) { case (innerProduct, (feature, count)) =>
                  val globalCount =
                    if (feature == -1) m.massTotal + { m match {
                      case withPrior: DirichletPrior if withPrior.prior != null => withPrior.prior.massTotal
                      case _ => 0.0
                    }}
                    else m.masses(feature) + { m match {
                      case withPrior: DirichletPrior if withPrior.prior != null => withPrior.prior(feature)
                      case _ => 0.0
                    }}
                  var p = 1.0
                  while (i < count) {
                    p *= (globalCount + i)
                    i += 1
                  }
                  if (feature < 0) innerProduct / p
                  else innerProduct * p
                }
            }
          }
          val pr = pValue * cValue * nCRPValue
          sum += pr
          distribution(i) = pr
        }
        i+=1
      }
      val selectedIdx = if (sum == 0) candidates(random.nextInt(distribution.length))
                        else candidates(cc.factorie.maths.nextDiscrete(distribution, sum)(random))
      var selected = v.domain.apply(selectedIdx)
      while(selected.depth < maxLevel) selected = v.domain.newTreePath(selected)
      v.set(selected)(null)
      collapsedFactors.foreach(_.updateCollapsedParents(1.0))
    }
  }

  class LevelClosure(vs:MutableSeqVar[MutableIntegerVar],sampler:CollapsedGibbsSampler, samplingCandidates: Seq[Int])(implicit random: scala.util.Random) extends CollapsedGibbsSamplerClosure {
    val model = sampler.model
    val (allCrpFactors, childFactors) = model.childFactors(vs).partition(_.isInstanceOf[PlatedNestedCRP.Factor])
    require(allCrpFactors.size <= 1)
    val parentFactor = model.getParentFactor(vs).get.asInstanceOf[PlatedStickBreakingProcess.Factor]
    val collapsedFactors = ArrayBuffer[DirectedFactor]()
    collapsedFactors ++= childFactors.filter(f => f.parents.exists(sampler.isCollapsed))
    if (parentFactor.parents.exists(sampler.isCollapsed))
      collapsedFactors += parentFactor

    val nCrpFactor = allCrpFactors.head.asInstanceOf[PlatedNestedCRP.Factor]

    //sample in 2 stages: 1st from all levels up to maximum level of current tree path + 1 level deeper -> if deeper is sampled, sample from bernoulli to go even deeper (create new paths if necessary)
    def sample(implicit d:DiffList = null) {
      val maxLevel = nCrpFactor.path.value.depth
      //pre-calculate path priors, because their calculation is recursive it is better to pre-calculate them
      val getParentScore:(Int)=>Double = {
          val props = parentFactor.probabilitiesForLevels(maxLevel)
          idx => props(parentFactor.child(idx).intValue)
      }
      val distribution = new Array[Int](maxLevel + 2)
      (0 until vs.length).foreach(idx => {
        updateCollapsedParents(-1.0,idx)
        val v = vs(idx)
        var candidateLevel = 0
        var childSum = 0.0
        while(candidateLevel < distribution.length - 1) {
          v.set(candidateLevel)(null)
          val cValue = childFactors.foldLeft(1.0) {
            case (acc: Double, f: SeqParentFactor) => acc * f.proportionalForParentIndex(idx)
            //Defaults that could potentially be very slow if we are sampling a SeqVar
            case (acc: Double, f: DirectedFactor) => acc * f.pr
          }
          val pValue = getParentScore(idx)
          val prob = cValue * pValue
          distribution(candidateLevel) = prob
          childSum+=cValue
          candidateLevel+=1
        }
        //divide distribution by childSum to get the real probability for each candidate depth
        (0 until distribution.length).foreach(i => distribution(i)/=childSum)
        //probability to go deeper is the leftover probability
        distribution(maxLevel+1) = 0.0
        distribution(maxLevel+1) = 1.0-distribution.sum
        var selectedLevel =
          if (sum == 0) random.nextInt(distribution.length)
          else cc.factorie.maths.nextDiscrete(distribution)(random)
        if(selectedLevel == maxLevel+1) {
          //Sample new paths depth with stick-breaking -> m*(1-m)^depth
          var pr = f.m.doubleValue
          var r = random.nextDouble()
          var selectedPath = nCrpFactor.path.domain.newTreePath(nCrpFactor.path.value)
          while(r > pr) {
            //Create a new path in the nCRP tree
            selectedPath = nCrpFactor.path.domain.newTreePath(selectedPath)
            nCrpFactor.path.set(selectedPath)(null)
            selectedLevel+=1
            r -= pr
            pr*=(1.0-pr)
          }
        }
        v.set(selectedLevel)(null)
        updateCollapsedParents(1.0,idx)
      })
    }
    def updateCollapsedParents(weight:Double, idx:Int) {
      collapsedFactors.foreach {
        case f: SeqGeneratingFactor if parentFactor == f => f.updateCollapsedParentsForIdx(weight, idx)
        case f: SeqParentFactor => f.updateCollapsedParentsForParentIdx(weight, idx)
        case f: DirectedFactor => throw new Error("Factors connected to sequence variables should implement either SeqGeneratingFactor (for parent factors of such) or SeqParentFactor (for child factors), to update collapsed parents efficiently!")
      }
    }
  }
}