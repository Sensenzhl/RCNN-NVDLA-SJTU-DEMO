#ifndef CAFFE_DATADISTR_LAYER_HPP_
#define CAFFE_DATADISTR_LAYER_HPP_

#include <vector>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"


namespace caffe {
#ifdef AMOD_CONFIG_ENA
extern string hist_dir;

template <typename Dtype>
class ConvolutionLayer;
template <typename Dtype>
class SplitCConvolutionLayer;
template <typename Dtype>
class WinogradConvolutionLayer;
template <typename Dtype>
class BaseConvolutionLayer;
template <typename Dtype>
class InnerProductLayer;
template <typename Dtype>
class NVLRNLayer;
template <typename Dtype>
class NVSigmoidLayer;
template <typename Dtype>
class NVTanHLayer;
template <typename Dtype>
class DataDistriWrapperLayer;
template <typename Dtype>
class Layer;

/**
 * @brief Creates a "DataDistribute" path in the network by take bottom blobs
 *        as input and calculate data distribution
 *
 */
template <typename Dtype>
class DataDistributeLayer {
 public:
     friend class ConvolutionLayer<Dtype>;
     friend class SplitCConvolutionLayer<Dtype>;
     friend class WinogradConvolutionLayer<Dtype>;
     friend class BaseConvolutionLayer<Dtype>;
     friend class InnerProductLayer<Dtype>;
     friend class NVLRNLayer<Dtype>;
     friend class NVSigmoidLayer<Dtype>;
     friend class NVTanHLayer<Dtype>;
     friend class DataDistriWrapperLayer<Dtype>;
     friend class Layer<Dtype>;
  explicit DataDistributeLayer(const DataDistributeParameter &param)
      : datadist_param_(param) {
          max_ = std::numeric_limits<Dtype>::min();
          min_ = std::numeric_limits<Dtype>::max();
          cfg_max_ = 0;
          cfg_min_ = 0;
          limit_ = 0;
          sum_ = 0;
          max_sum_weight_ = 0;
          mute_ = false;
          set_datadist_param( param );
          reset_hist();
      }

  inline void set_datadist_param( const DataDistributeParameter &param ) {
      datadist_param_ = param;

      num_marker_         = datadist_param_.num_marker();
      cfg_max_            = datadist_param_.max_limit();
      cfg_min_            = datadist_param_.min_limit();
      // FIXME: adding limit
      limit_              = std::max(fabs(cfg_max_), fabs(cfg_min_));
      scale_              = 2*limit_/(num_marker_-1);
      CHECK_EQ( num_marker_%2, 1 ) << "num_marker has to be odds";

      linear_marker_.resize(num_marker_, 0);
      hist_linear_.resize(num_marker_, 0);
      exp_marker_.resize(num_marker_, 0);
      hist_exp_.resize(num_marker_, 0);

      linear_marker_[0] = -limit_;
      for( int i = 1; i < num_marker_; i++ )
      {
          linear_marker_[i] = -limit_ + i*scale_;
      }

      // Create data distribute markers based on user configuration
      double mark;
      mark = std::max(fabs(cfg_min_), fabs(cfg_max_));
      exp_marker_[num_marker_/2] = 0;
      for( int i = num_marker_/2; i > 1; i-- )
      {
          exp_marker_[num_marker_/2+i] = mark;
          exp_marker_[num_marker_/2-i] = -mark;
          mark /= 2;
      }
      count_ = 0;
  }
  void set_layer_name(const string layer_name ) {
    layer_name_ = layer_name;
  }
  inline void reset_hist( ) {
      for( int i = 0; i < num_marker_; i++ ) {
        hist_linear_[i] = 0;
        hist_exp_[i] = 0;
      }
      count_ = 0;
  }
  inline void set_mute( bool mute ) { mute_ = mute; }

  void tell( const string filename) {
      DataDistributeProto proto;
      ToProto( &proto );
      WriteProtoToTextFile( proto, filename.c_str());
  }

  void Forward_cpu(const vector<Blob<Dtype>*>& bottom );
  void Forward_weight(const vector<Blob<Dtype>*>& bottom );

 private:
  void ToProto(DataDistributeProto *proto );


    DataDistributeParameter datadist_param_;

    bool                mute_;

    // From configuration
    int                 num_marker_;
    double              scale_;

    // Stats output of this layer
    Dtype               max_;
    Dtype               min_;
    Dtype               cfg_max_;
    Dtype               cfg_min_;
    Dtype               limit_;
    Dtype               sum_;
    Dtype               max_sum_weight_;
    vector<Dtype>       linear_marker_;
    vector<uint64_t>    hist_linear_;
    vector<Dtype>       exp_marker_;
    vector<uint64_t>    hist_exp_;
    uint64_t            count_;

    string              layer_name_;
};
#endif



}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_
