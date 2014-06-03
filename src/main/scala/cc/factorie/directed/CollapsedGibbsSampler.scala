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
import scala.collection.mutable.{ArrayBuffer, HashSet}
import cc.factorie.variable._
import cc.factorie.directed.factor.{DiscreteSeqGeneratingFactor, PlatedDiscreteMixture, PlatedDiscrete, DirectedFactor}
import scala.collection.mutable
import cc.factorie.util.FastLogging

/**
 *
 * @param collapse variables that are being collapsed
 * @param model
 * @param samplingCandidates A function that assigns candidates for sampling to a variable (Optional).
 *                           If it returns null for a variable, the whole domain is a possible candidates (default, can be slow).
 *                           For SeqVars this function should return candidates for the elements of the SeqVars!
 * @param random
 */
class CollapsedGibbsSampler(collapse: Iterable[Var], val model: DirectedModel, samplingCandidates: Var => Seq[Int] = _ => null)(implicit val random: scala.util.Random)
  extends Sampler[Iterable[MutableVar]]
  with FastLogging {

  var debug = false
  makeNewDiffList = false
  var temperature = 1.0

  private val collapsed = new HashSet[Var] ++ collapse

  // Initialize collapsed parameters specified in constructor
  val collapser = new Collapse(model)
  collapse.foreach(v => collapser(Seq(v)))

  def isCollapsed(v: Var): Boolean = collapsed.contains(v)

  def process1(vs: Iterable[MutableVar]): DiffList = {
    val d = newDiffList

    //Cache for counts; important to make for example PlatedMultinomialFromSeq.Factor sampling faster.
    // Otherwise iterating over the whole SeqVar would be necessary for each sampling candidate
    val countsCache = mutable.HashMap[DiscreteSeqVar,mutable.HashMap[Int,Double]]()
    def getCount(variable:DiscreteSeqVar, candidate:Int) =
      countsCache.getOrElseUpdate(variable,
        variable.foldLeft(mutable.HashMap[Int, Double]()){ case (counts, value) => counts += value.intValue -> (1.0 + counts.getOrElse(value.intValue, 0.0)); counts}
      ).getOrElse(candidate,0.0)

    vs.foreach {
      //Different sampling for Vars and SeqVars
      case v: MutableDiscreteVar =>
        val childFactors = model.childFactors(v)
        val parentFactor = model.getParentFactor(v)

        val collapsedFactors = ArrayBuffer[DirectedFactor]()
        collapsedFactors ++= childFactors.filter(f => f.parents.exists(isCollapsed))
        if (parentFactor.isDefined && parentFactor.get.parents.exists(isCollapsed))
          collapsedFactors += parentFactor.get

        collapsedFactors.foreach(_.updateCollapsedParents(-1.0))

        var sum = 0.0

        var candidates = samplingCandidates(v)
        if (candidates == null)
          candidates = 0 until v.domain.size

        val distribution = Array.ofDim[Double](candidates.size)
        (0 until distribution.length).foreach {
          idx =>
            val candidate = candidates(idx)
            v.set(candidate)(null)
            val pValue = parentFactor match {
              case Some(f:MultinomialFromSeq.Factor) => getCount(f._2,candidate) //efficiently cache element counts of parent
              case Some(f:DirectedFactor) => f.pr
              case None => 1.0
            }
            val cValue = childFactors.foldLeft(1.0)(_ * _.pr)

            val pr = pValue * cValue
            sum += pr
            distribution(idx) = pr
        }

        if (sum == 0) v.set(candidates(random.nextInt(distribution.length)))(null)
        else v.set(candidates(cc.factorie.maths.nextDiscrete(distribution, sum)(random)))(null)

        collapsedFactors.foreach(_.updateCollapsedParents(1.0))

      case v: MutableDiscreteSeqVar[_] if !v.isEmpty =>
        val childFactors = model.childFactors(v)
        val parentFactor = model.getParentFactor(v)

        var candidates = samplingCandidates(v)
        if (candidates == null)
          candidates = 0 until v.domain.elementDomain.size

        val collapsedFactors = ArrayBuffer[DiscreteSeqGeneratingFactor]()
        collapsedFactors ++= childFactors.filter(f => f.parents.exists(isCollapsed)).map(_.asInstanceOf[DiscreteSeqGeneratingFactor])
        if (parentFactor.isDefined && parentFactor.get.parents.exists(isCollapsed))
          collapsedFactors += parentFactor.get.asInstanceOf[DiscreteSeqGeneratingFactor]

        //HACK: Init cached counts for v if childFactors contain [Plated]MultinomialFromSeq.Factor, to be sure that current counts are being cached
        if(childFactors.exists(f => f.isInstanceOf[PlatedMultinomialFromSeq.Factor] || f.isInstanceOf[MultinomialFromSeq.Factor]))
          getCount(v,0)

        (0 until v.size).foreach(idx => {
          var sum = 0.0
          val distribution = Array.ofDim[Double](candidates.length)

          //update sufficient statistics
          collapsedFactors.foreach(f => f.updateCollapsedParentsForIdx(-1.0, idx))
          //update cache if existing, decrement current element
          countsCache.get(v).foreach(_(v.intValue(idx)) -= 1)

          (0 until candidates.size).foreach {
            i =>
              val candidate = candidates(i)
              v.set(idx, candidate)(null)

              val pValue = parentFactor match {
                case Some(f: PlatedMultinomialFromSeq.Factor) => getCount(f._2, candidate)  //efficient by using cache
                //Fast because it only calculates probability for current index
                case Some(f: DiscreteSeqGeneratingFactor) => f.prForIndex(idx)
                //Defaults that could potentially be very slow if we are sampling a SeqVar (e.g., above two cases)
                case Some(f: DirectedFactor) =>
                  logger.warn(s"Sampling DiscreteSeqVar using ${f.getClass} could be slow. ${f.getClass} should inherit from DiscreteSeqGeneratingFactor!")
                  f.pr
                case None => 1.0
              }

              val cValue = childFactors.foldLeft(1.0) {
                //make this faster by not calculating the whole probability here, because only the variable at idx changes
                case (acc: Double, f: PlatedDiscreteMixture.Factor) => acc * f._2(candidate).value(f._1(idx).intValue)
                case (acc: Double, f: MultinomialFromSeq.Factor) => acc * {
                  if(f._1.intValue == candidate) {
                    //efficient through caching
                    val ct = getCount(f._2, candidate)
                    (ct+1.0)/ct
                  } else 1.0
                }
                case (acc: Double, f: PlatedMultinomialFromSeq.Factor) => acc * {
                  //efficient through caching
                  val parentCount = getCount(f._2, candidate)
                  val childCount = getCount(f._1, candidate)
                  math.pow((parentCount+1.0)/parentCount,childCount)
                }
                //Defaults that could potentially be very slow if we are sampling a SeqVar (e.g., above two cases)
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

          //update cache if existing; increment count of selected element
          countsCache.get(v).foreach(m => m += selected ->  (1.0 + m.getOrElse(selected,0.0)))
          //update sufficient statistics
          collapsedFactors.foreach(f => f.updateCollapsedParentsForIdx(1.0, idx))
        })

      case _ => throw new IllegalArgumentException("Can only sample for DiscreteVar or DiscreteSeqVar at the moment!")
    }

    d
  }

  /** Convenience for sampling single variable */
  def process(v: MutableVar): DiffList = process(Seq(v))

}