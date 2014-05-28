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
import scala.collection.mutable.{HashSet}
import cc.factorie.variable._

/** A GibbsSampler that can also collapse some Parameters. */
class CollapsedGibbsSampler(collapse:Iterable[Var], val model:DirectedModel)(implicit val random: scala.util.Random) extends Sampler[Iterable[MutableVar]] {
  var debug = false
  makeNewDiffList = false // override default in cc.factorie.Sampler
  var temperature = 1.0 // TODO Currently ignored?

  private val collapsed = new HashSet[Var] ++ collapse

  // Initialize collapsed parameters specified in constructor
  val collapser = new Collapse(model)
  collapse.foreach(v => collapser(Seq(v)))
  // TODO We should provide an interface that handlers can use to query whether or not a particular variable was collapsed or not?

  def isCollapsed(v:Var): Boolean = collapsed.contains(v)
  
  def process1(vs:Iterable[MutableVar]): DiffList = {
    val d = newDiffList

    vs.foreach{
      case v:MutableDiscreteVar =>
        val childFactors = model.childFactors(v)
        val parentFactor = model.parentFactor(v)

        val domainSize = v.domain.size

        childFactors.map(f => if(f.parents.headOption.exists(isCollapsed)) f.updateCollapsedParents(-1.0))
        if(parentFactor.parents.headOption.exists(isCollapsed)) parentFactor.updateCollapsedParents(-1.0)

        var sum = 0.0
        val distribution = Array.ofDim[Double](domainSize)
        (0 until domainSize).foreach { value1 =>
          val value = v.domain(value1)

          val pValue = parentFactor match {
            case f:DirectedFactorWithStatistics3[_,_,_] => f.pr(value.asInstanceOf[f._1.Value], f._2.value, f._3.value)
            case f:DirectedFactorWithStatistics2[_,_] => f.pr(value.asInstanceOf[f._1.Value], f._2.value)
            case f:DirectedFactorWithStatistics1[_] => f.pr(value.asInstanceOf[f._1.Value])
          }

          val cValue =  childFactors.foldLeft(0.0) {
            case (acc:Double,f:DirectedFactorWithStatistics2[_,_]) => acc * f.pr(f._1.value, value.asInstanceOf[f._2.Value])
            case (acc:Double,f:DirectedFactorWithStatistics3[_,_,_]) => acc * f.pr(f._1.value, f._2.value, value.asInstanceOf[f._3.Value])
          }

          val pr = pValue * cValue
          sum += pr
          distribution(value1) = pr
        }

        if (sum == 0) v.set(random.nextInt(domainSize))(null)
        else v.set(cc.factorie.maths.nextDiscrete(distribution, sum)(random))(null)

        childFactors.map(f => if(f.parents.exists(isCollapsed)) f.updateCollapsedParents(1.0))
        if(parentFactor.parents.headOption.exists(isCollapsed)) parentFactor.updateCollapsedParents(1.0)

      case v:MutableDiscreteSeqVar[_] if !v.isEmpty =>       //TODO: make this more efficient
        val childFactors = model.childFactors(v)
        val parentFactor = model.parentFactor(v)

        val domainSize = v.domain.elementDomain.size

        (0 until v.size).foreach(idx => {
          var sum = 0.0
          val distribution = Array.ofDim[Double](domainSize)

          childFactors.map(f => if(f.parents.exists(isCollapsed)) f.updateCollapsedParentsForIdx(-1.0,idx))
          if(parentFactor.parents.headOption.exists(isCollapsed)) parentFactor.updateCollapsedParentsForIdx(-1.0,idx)

          (0 until domainSize).foreach { value1 =>
            lazy val value = v.value.updated(idx,v.domain.elementDomain(value1))  //don't compute if you do not have to

            val pValue = parentFactor match {
              //make this faster by not calculating the whole probability here, because only the variable at idx changes
              case f:PlatedDiscreteMixture.Factor => f._2(f._1(idx).intValue).value(value1)
              case f:PlatedDiscrete.Factor => f._2.value(value1)
              //Defaults that could potentially be very slow if we are sampling a SeqVar (e.g., above two cases)
              case f:DirectedFactorWithStatistics3[_,_,_] => f.pr(value.asInstanceOf[f._1.Value], f._2.value, f._3.value)
              case f:DirectedFactorWithStatistics2[_,_] => f.pr(value.asInstanceOf[f._1.Value], f._2.value)
              case f:DirectedFactorWithStatistics1[_] => f.pr(value.asInstanceOf[f._1.Value])
            }

            val cValue =  childFactors.foldLeft(1.0) {
              //make this faster by not calculating the whole probability here, because only the variable at idx changes
              case (acc:Double,f:PlatedDiscreteMixture.Factor) => acc * f._2(value1).value(f._1(idx).intValue)
              //Defaults that could potentially be very slow if we are sampling a SeqVar (e.g., above two cases)
              case (acc:Double,f:DirectedFactorWithStatistics2[_,_]) => acc * f.pr(f.child.value, value.asInstanceOf[f._2.Value])
              case (acc:Double,f:DirectedFactorWithStatistics3[_,_,_]) => acc * f.pr(f.child.value, f._2.value, value.asInstanceOf[f._3.Value])
            }

            val pr = pValue * cValue
            sum += pr
            distribution(value1) = pr
          }
          
          if (sum == 0) v.set(idx, random.nextInt(domainSize))(null)
          else {
            v.set(idx, cc.factorie.maths.nextDiscrete(distribution, sum)(random))(null)
          }

          childFactors.map(f => if(f.parents.exists(isCollapsed)) f.updateCollapsedParentsForIdx(1.0,idx))
          if(parentFactor.parents.exists(isCollapsed)) parentFactor.updateCollapsedParentsForIdx(1.0,idx)
        })

      case _ => throw new IllegalArgumentException("Can only sample for DiscreteVar or DiscreteSeqVar")
    }

    d
  }

  /** Convenience for sampling single variable */
  def process(v:MutableVar): DiffList = process(Seq(v))

}