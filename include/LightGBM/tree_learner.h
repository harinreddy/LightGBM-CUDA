/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_TREE_LEARNER_H_
#define LIGHTGBM_TREE_LEARNER_H_

#include <LightGBM/config.h>
#include <LightGBM/meta.h>

#include <string>
#include <vector>

#include <LightGBM/json11.hpp>

using namespace json11;

namespace LightGBM {

/*! \brief forward declaration */
class Tree;
class Dataset;
class ObjectiveFunction;

/*!
* \brief Interface for tree learner
*/
class TreeLearner {
 public:
  /*! \brief virtual destructor */
  virtual ~TreeLearner() {}

  /*!
  * \brief Initialize tree learner with training dataset
  * \param train_data The used training data
  * \param is_constant_hessian True if all hessians share the same value
  */
  // ibmGBT: added is_use_subset_ for CUDA
  virtual void Init(const Dataset* train_data, bool is_constant_hessian, bool is_use_subset ) = 0;

  virtual void ResetTrainingData(const Dataset* train_data) = 0; 

  /*!
  * \brief Reset tree configs
  * \param config config of tree
  */
  virtual void ResetConfig(const Config* config) = 0;

  /*!
  * \brief training tree model on dataset 
  * \param gradients The first order gradients
  * \param hessians The second order gradients
  * \param is_constant_hessian True if all hessians share the same value
  * \return A trained tree
  */
  virtual Tree* Train(const score_t* gradients, const score_t* hessians, bool is_constant_hessian,
                      Json& forced_split_json) = 0;

  /*!
  * \brief use a existing tree to fit the new gradients and hessians.
  */
  virtual Tree* FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) const = 0;

  virtual Tree* FitByExistingTree(const Tree* old_tree, const std::vector<int>& leaf_pred,
                                  const score_t* gradients, const score_t* hessians) = 0;

  /*!
  * \brief Set bagging data
  * \param used_indices Used data indices
  * \param num_data Number of used data
  */
  virtual void SetBaggingData(const data_size_t* used_indices,
    data_size_t num_data) = 0;

  /*!
  * \brief Using last trained tree to predict score then adding to out_score;
  * \param out_score output score
  */
  virtual void AddPredictionToScore(const Tree* tree, double* out_score) const = 0;

  virtual void RenewTreeOutput(Tree* tree, const ObjectiveFunction* obj, std::function<double(const label_t*, int)> residual_getter,
                               data_size_t total_num_data, const data_size_t* bag_indices, data_size_t bag_cnt) const = 0;

  TreeLearner() = default;
  /*! \brief Disable copy */
  TreeLearner& operator=(const TreeLearner&) = delete;
  /*! \brief Disable copy */
  TreeLearner(const TreeLearner&) = delete;

  /*!
  * \brief Create object of tree learner
  * \param learner_type Type of tree learner
  * \param device_type Type of tree learner
  * \param config config of tree
  */
  static TreeLearner* CreateTreeLearner(const std::string& learner_type,
    const std::string& device_type,
    const Config* config);

  // ibmGBT
  // return timing result
  virtual std::chrono::duration<double, std::milli> GetInitTrainTime() = 0; 
  virtual std::chrono::duration<double, std::milli> GetInitSplitTime() = 0; 
  virtual std::chrono::duration<double, std::milli> GetHistTime() = 0; 
  virtual std::chrono::duration<double, std::milli> GetFindSplitTime() = 0; 
  virtual std::chrono::duration<double, std::milli> GetSplitTime() = 0; 
};

}  // namespace LightGBM

#endif   // LightGBM_TREE_LEARNER_H_
