#ifndef CAFFE_LOADER_LAYER_HPP_
#define CAFFE_LOADER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ConvolutionLayer;
template <typename Dtype>
class SplitCConvolutionLayer;
template <typename Dtype>
class WinogradConvolutionLayer;
template <typename Dtype>
class BaseConvolutionLayer;
template <typename Dtype>
class DeconvolutionLayer;
template <typename Dtype>
class DataDistriWrapperLayer;
template <typename Dtype>
class InnerProductLayer;
/**
 * @brief Creates a "loader" path in the network so that we can load
 *      the intermidate to support Comparison
 *
 * TODO(dox): add more paramter and functionality
 */
template <typename Dtype>
class LoaderLayer: public Layer<Dtype> {
 public:
     friend class ConvolutionLayer<Dtype>;
     friend class SplitCConvolutionLayer<Dtype>;
     friend class WinogradConvolutionLayer<Dtype>;
     friend class BaseConvolutionLayer<Dtype>;
     friend class DeconvolutionLayer<Dtype>;
     friend class DataDistriWrapperLayer<Dtype>;
     friend class InnerProductLayer<Dtype>;
  explicit LoaderLayer(const LayerParameter& param)
      : Layer<Dtype>(param){
          loader_param = param.loader_param();
      }
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Loader"; }
  virtual inline int MinBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline bool NeedDumpTopBlobs() const { return false; }
  inline void set_loader_param( const LoaderParameter &param ) {
      loader_param = param;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 private:
  LoaderParameter   loader_param;
};

}  // namespace caffe

#endif  // CAFFE_INNER_PRODUCT_LAYER_HPP_
