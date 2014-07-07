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

import cc.factorie.infer._
import cc.factorie.util.FastLogging
import scala.collection.mutable.{HashMap, HashSet, ArrayBuffer}
import cc.factorie.variable._

/** A GibbsSampler that can also collapse some Parameters. */
class CollapsedGibbsSampler(collapse:Iterable[Var], val model:DirectedModel)(implicit val random: scala.util.Random) extends Sampler[Iterable[MutableVar]] {
  var debug = false
  makeNewDiffList = false // override default in cc.factorie.Sampler
  var temperature = 1.0 // TODO Currently ignored?
  val handlers = new ArrayBuffer[CollapsedGibbsSamplerHandler]
  def defaultHandlers = Seq(
      new DefaultDiscreteGibbsSamplerHandler,
      GeneratedVarCollapsedGibbsSamplerHandler
      )
  handlers ++= defaultHandlers
  val cacheClosures = true
  val closures = new HashMap[Var, CollapsedGibbsSamplerClosure]
  private val collapsed = new HashSet[Var] ++ collapse

  // Initialize collapsed parameters specified in constructor
  val collapser = new Collapse(model)
  collapse.foreach(v => collapser(Seq(v)))
  // TODO We should provide an interface that handlers can use to query whether or not a particular variable was collapsed or not?

  def isCollapsed(v:Var): Boolean = collapsed.contains(v)
  
  def process1(v:Iterable[MutableVar]): DiffList = {
    //assert(!v.exists(_.isInstanceOf[CollapsedVar])) // We should never be sampling a CollapsedVariable
    val d = newDiffList
    // If we have a cached closure, just use it and return
    if (cacheClosures && v.size == 1 && closures.contains(v.head)) { 
      closures(v.head).sample(d)
    } else {
      //println("CollapsedGibbsSampler.process1 factors = "+factors.map(_.template.getClass).mkString)
      var done = false
      val handlerIterator = handlers.iterator
      while (!done && handlerIterator.hasNext) {
        val closure = handlerIterator.next().sampler(v, this)
        if (closure ne null) {
          done = true
          closure.sample(d)
          if (cacheClosures && v.size == 1) {
            closures(v.head) = closure
          }
        }
      }
      if (!done) throw new Error("CollapsedGibbsSampler: No sampling method found for variable "+v+" with factors "+model.factors(v).map(_.factorName).toList.mkString)
    }
    d
  }

  /** Convenience for sampling single variable */
  def process(v:MutableVar): DiffList = process(Seq(v))

}


trait CollapsedGibbsSamplerHandler {
  def sampler(v:Iterable[Var], sampler:CollapsedGibbsSampler)(implicit random: scala.util.Random): CollapsedGibbsSamplerClosure
}

trait CollapsedGibbsSamplerClosure {
  def sample(implicit d:DiffList = null): Unit
}

//Mixin for handlers to provide special handling for sampling candidates
trait SelectedCandidatesCollapsedGibbsSamplerHandler {
  def samplingCandidates(v:MutableVar):Seq[v.Value]
}
//Mixin for handlers to provide special handling for sampling candidates, for SeqVar
trait PlatedSelectedCandidatesCollapsedGibbsSamplerHandler {
  def elementSamplingCandidates[E](v:SeqVar[E]):Seq[E]
}

object GeneratedVarCollapsedGibbsSamplerHandler extends CollapsedGibbsSamplerHandler {
  def sampler(v:Iterable[Var], sampler:CollapsedGibbsSampler)(implicit random: scala.util.Random): CollapsedGibbsSamplerClosure = {
    val factors = sampler.model.factors(v)
    if (v.size != 1 || factors.size != 1) return null
    val pFactor = factors.collectFirst({case f:DirectedFactor => f}) // TODO Yipes!  Clean up these tests!
    if (pFactor == None) return null
    // Make sure all parents are collapsed?
    //if (!pFactor.get.variables.drop(1).asInstanceOf[Seq[Parameter]].forall(v => sampler.collapsedMap.contains(v))) return null
    new Closure(pFactor.get)
  }
  class Closure(val factor:DirectedFactor)(implicit random: scala.util.Random) extends CollapsedGibbsSamplerClosure {
    def sample(implicit d:DiffList = null): Unit = {
      factor.updateCollapsedParents(-1.0)
      val variable = factor.child.asInstanceOf[MutableVar]
      variable.set(factor.sampledValue.asInstanceOf[variable.Value])
      factor.updateCollapsedParents(1.0)
      // TODO Consider whether we should be passing values rather than variables to updateChildStats
      // TODO What about collapsed children?
    }
  }
}

//Proportionals of candidates for sampling are products of all neighbouring factor proportionals.
//Can basically handle all discrete variables and factors, though some factors might not be very efficient.
//Mixin traits SelectedCandidatesCollapsedGibbsSamplerHandler or PlatedSelectedCandidatesCollapsedGibbsSamplerHandler can be used for special candidate selection.
class DefaultDiscreteGibbsSamplerHandler extends CollapsedGibbsSamplerHandler {
  self =>
  def sampler(vs:Iterable[Var], sampler:CollapsedGibbsSampler)(implicit random: scala.util.Random) = {
    try {
      if(vs.size != 1) null
      else
        vs.head match {
          case v:MutableDiscreteVar => new DiscreteClosure(v,sampler)
          case v:MutableDiscreteSeqVar[_] => new DiscreteSeqClosure(v,sampler)
          case _ => null
        }
    } catch {
      case e:java.lang.ClassCastException => throw new ClassCastException("Sampling candidates of DiscreteVar have to be of type DiscreteValue")
    }
  }
  class DiscreteClosure(v:MutableDiscreteVar,sampler:CollapsedGibbsSampler)(implicit random: scala.util.Random) extends CollapsedGibbsSamplerClosure {
    val model = sampler.model
    val childFactors = model.childFactors(v)
    val parentFactor = model.getParentFactor(v)
    val collapsedFactors = ArrayBuffer[DirectedFactor]()
    collapsedFactors ++= childFactors.filter(f => f.parents.exists(sampler.isCollapsed))
    if (parentFactor.isDefined && parentFactor.get.parents.exists(sampler.isCollapsed))
      collapsedFactors += parentFactor.get

    def sample(implicit d:DiffList = null) {
      val candidates:Seq[Int] = self match {
        case h:SelectedCandidatesCollapsedGibbsSamplerHandler => val cands = h.samplingCandidates(v); if(cands != null) cands.map(_.intValue) else 0 until v.domain.size
        case _ => 0 until v.domain.size
      }
      val distribution = Array.ofDim[Double](candidates.size)
      collapsedFactors.foreach(_.updateCollapsedParents(-1.0))
      var sum = 0.0
      (0 until distribution.length).foreach {
        i =>
          val candidate = candidates(i)
          v.set(candidate)(null)
          val pValue = parentFactor.fold(1.0)(_.pr)
          val cValue = childFactors.foldLeft(1.0)(_ * _.pr)
          val pr = pValue * cValue
          sum += pr
          distribution(i) = pr
      }
      if (sum == 0) v.set(candidates(random.nextInt(distribution.length)))(null)
      else v.set(candidates(cc.factorie.maths.nextDiscrete(distribution, sum)(random)))(null)
      collapsedFactors.foreach(_.updateCollapsedParents(1.0))
    }
  }
  class DiscreteSeqClosure(v:MutableDiscreteSeqVar[_],sampler:CollapsedGibbsSampler)(implicit random: scala.util.Random) extends CollapsedGibbsSamplerClosure with FastLogging {
    val model = sampler.model
    val childFactors = model.childFactors(v)
    val parentFactor = model.getParentFactor(v)
    val collapsedFactors = ArrayBuffer[DirectedFactor]()
    collapsedFactors ++= childFactors.filter(f => f.parents.exists(sampler.isCollapsed))
    if (parentFactor.isDefined && parentFactor.get.parents.exists(sampler.isCollapsed))
      collapsedFactors += parentFactor.get

    def sample(implicit d:DiffList = null) {
      val candidates:Seq[Int] = self match {
        case h:PlatedSelectedCandidatesCollapsedGibbsSamplerHandler => val cands = h.elementSamplingCandidates(v); if(cands != null) cands.map(_.intValue) else 0 until v.domain.elementDomain.size
        case _ => 0 until v.domain.elementDomain.size
      }
      val distribution = Array.ofDim[Double](candidates.size)
      (0 until v.size).foreach(index => {
        var sum = 0.0
        //update sufficient statistics
        updateCollapsedParents(-1.0,index)
        (0 until distribution.length).foreach {
          i =>
            val candidate = candidates(i)
            v.set(index, candidate)(null)
            val pValue = parentFactor match {
              //Fast because it only calculates probability for current index
              case Some(f: SeqGeneratingFactor) => f.proportionalForChildIndex(index)
              //Defaults that could potentially be very slow if we are sampling a SeqVar (e.g., above two cases)
              case Some(f: DirectedFactor) =>
                logger.warn(s"Sampling DiscreteSeqVar using ${f.getClass} could be slow. ${f.getClass} should inherit from DiscreteSeqGeneratingFactor!")
                f.pr
              case None => 1.0
            }
            val cValue = childFactors.foldLeft(1.0) {
              //useful for PlatedDiscreteMixture
              case (acc: Double, f: SeqAsParentFactor) => acc * f.proportionalForParentIndex(index)
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
        v.set(index,selected)(null)
        //update sufficient statistics
        updateCollapsedParents(1.0,index)
      })
    }
    def updateCollapsedParents(weight:Double, idx:Int) {
      collapsedFactors.foreach {
        case f: SeqGeneratingFactor if parentFactor.exists(_ == f) => f.updateCollapsedParentsForIdx(weight, idx)
        case f: SeqAsParentFactor => f.updateCollapsedParentsForParentIdx(weight, idx)
        case f: DirectedFactor => throw new Error("Factors connected to discrete sequence variables should implement either SeqGeneratingFactor (for parent factors of such) or SeqParentFactor (for child factors), to update collapsed parents efficiently!")
      }
    }
  }
}
