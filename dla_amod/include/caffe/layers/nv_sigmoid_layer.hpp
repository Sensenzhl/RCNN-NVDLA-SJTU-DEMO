#ifndef CAFFE_NV_SIGMOID_LAYER_HPP_
#define CAFFE_NV_SIGMOID_LAYER_HPP_

#include <vector>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/lut_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"

namespace caffe {
#ifdef AMOD_CONFIG_ENA
/**
 * @brief NVSigmoid function non-linearity @f$
 *         y = (1 + \exp(-x))^{-1}
 *     @f$, a classic choice in neural networks.
 *
 * Note that the gradient vanishes as the values move away from 0.
 * The ReLULayer is often a better choice for this reason.
 */
template <typename Dtype>
class NVSigmoidLayer : public NeuronLayer<Dtype> {
 public:
  explicit NVSigmoidLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param), lut_(param) { }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "NVSigmoid"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *        y = (1 + \exp(-x))^{-1}
   *      @f$
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the NVSigmoid inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (N \times C \times H \times W) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to computed outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial x}
   *            = \frac{\partial E}{\partial y} y (1 - y)
   *      @f$ if propagate_down[0]
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  LUTLayer<Dtype>                  lut_;
};
#endif
}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
