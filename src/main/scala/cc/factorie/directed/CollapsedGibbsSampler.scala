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
import cc.factorie.directed.factor.{PlatedDiscreteMixture, PlatedDiscrete, DirectedFactor}

/**
 *
 * @param collapse variables that are being collapsed
 * @param model
 * @param samplingCandidates A function that assigns candidates for sampling to a variable (Optional).
 *                           If it returns null for a variable, the whole domain is a possible candidates (default, can be slow).
 *                           Does not work for SeqVars at the moment!
 * @param random
 */
class CollapsedGibbsSampler(collapse: Iterable[Var], val model: DirectedModel, samplingCandidates: Var => Seq[Int] = _ => null)(implicit val random: scala.util.Random) extends Sampler[Iterable[MutableVar]] {
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

    vs.foreach {
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
            val value1 = candidates(idx)
            v.set(value1)(null)
            val pValue = parentFactor.map(_.pr).getOrElse(1.0)
            val cValue = childFactors.foldLeft(1.0)(_ * _.pr)

            val pr = pValue * cValue
            sum += pr
            distribution(idx) = pr
        }

        if (sum == 0) v.set(candidates(random.nextInt(distribution.length)))(null)
        else v.set(candidates(cc.factorie.maths.nextDiscrete(distribution, sum)(random)))(null)

        collapsedFactors.foreach(_.updateCollapsedParents(1.0))

      case v: MutableDiscreteSeqVar[_] if !v.isEmpty => //TODO: make this more efficient
        val childFactors = model.childFactors(v)
        val parentFactor = model.getParentFactor(v)

        val domainSize = v.domain.elementDomain.size

        val collapsedFactors = ArrayBuffer[DirectedFactor]()
        collapsedFactors ++= childFactors.filter(f => f.parents.exists(isCollapsed))
        if (parentFactor.isDefined && parentFactor.get.parents.exists(isCollapsed))
          collapsedFactors += parentFactor.get

        (0 until v.size).foreach(idx => {
          var sum = 0.0
          val distribution = Array.ofDim[Double](domainSize)

          collapsedFactors.foreach(f => f.updateCollapsedParentsForIdx(-1.0, idx))

          (0 until domainSize).foreach {
            value1 =>
              v.set(idx, value1)(null)

              val pValue = parentFactor match {
                //faster by not calculating the whole probability here, because only the variable at idx changes
                case Some(f: PlatedDiscreteMixture.Factor) => f._2(f._3(idx).intValue).value(value1)
                case Some(f: PlatedDiscrete.Factor) => f._2.value(value1)
                //Defaults that could potentially be very slow if we are sampling a SeqVar (e.g., above two cases)
                case Some(f: DirectedFactor) => f.pr
                case None => 1.0
              }

              val cValue = childFactors.foldLeft(1.0) {
                //make this faster by not calculating the whole probability here, because only the variable at idx changes
                case (acc: Double, f: PlatedDiscreteMixture.Factor) => acc * f._2(value1).value(f._1(idx).intValue)
                //Defaults that could potentially be very slow if we are sampling a SeqVar (e.g., above two cases)
                case (acc: Double, f: DirectedFactor) => acc * f.pr
              }

              val pr = pValue * cValue
              sum += pr
              distribution(value1) = pr
          }

          if (sum == 0) v.set(idx, random.nextInt(domainSize))(null)
          else v.set(idx, cc.factorie.maths.nextDiscrete(distribution, sum)(random))(null)

          collapsedFactors.foreach(f => f.updateCollapsedParentsForIdx(1.0, idx))
        })

      case _ => throw new IllegalArgumentException("Can only sample for DiscreteVar or DiscreteSeqVar")
    }

    d
  }

  /** Convenience for sampling single variable */
  def process(v: MutableVar): DiffList = process(Seq(v))

}