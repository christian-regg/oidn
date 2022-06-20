// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

namespace oidn {

  // Tensor memory layout
  enum class TensorLayout
  {
    x,
    
    chw,
    Chw8c,  // blocked
    Chw16c, // blocked
    oihw,
    OIhw8i8o,   // blocked
    OIhw16i16o, // blocked

    hwc,
    ohwi,
  };

  template<TensorLayout layout>
  struct TensorLayoutTraits;

  template<>
  struct TensorLayoutTraits<TensorLayout::chw>
  {
    template<typename T>
    struct Addressing
    {
      static constexpr size_t wStride = sizeof(T);
      size_t hStride;
      size_t cStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int C, int H, int W)
      {
        hStride = size_t(W) * wStride;
        cStride = size_t(H) * hStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int c, int h, int w) const
      {
        return size_t(c) * cStride +
               size_t(h) * hStride +
               size_t(w) * wStride;
      }
    };
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::hwc>
  {
    template<typename T>
    struct Addressing
    {
      static constexpr size_t cStride = sizeof(T);
      size_t wStride;
      size_t hStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int C, int H, int W)
      {
        wStride = size_t(C) * cStride;
        hStride = size_t(W) * wStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int c, int h, int w) const
      {
        return size_t(c) * cStride +
               size_t(h) * hStride +
               size_t(w) * wStride;
      }
    };
  };

  template<typename T, int blockSize>
  struct TensorAddressingChwBc
  {
    static constexpr int B = blockSize;

    static constexpr size_t wStride = B * sizeof(T);
    size_t hStride;
    size_t CStride;

    TensorAddressingChwBc() = default;

    OIDN_HOST_DEVICE_INLINE TensorAddressingChwBc(int C, int H, int W)
    {
      hStride = size_t(W) * wStride;
      CStride = size_t(H) * hStride;
    }

    OIDN_HOST_DEVICE_INLINE size_t getOffset(int c, int h, int w) const
    {
      return size_t(c/B) * CStride +
             size_t(h)   * hStride +
             size_t(w)   * wStride +
             size_t(c%B) * sizeof(T);
    }
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::Chw8c>
  {
    template<typename T>
    using Addressing = TensorAddressingChwBc<T, 8>;
  };
  
  template<>
  struct TensorLayoutTraits<TensorLayout::Chw16c>
  {
    template<typename T>
    using Addressing = TensorAddressingChwBc<T, 16>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::oihw>
  {
    template<typename T>
    struct Addressing
    {
      static constexpr size_t wStride = sizeof(T);
      size_t hStride;
      size_t iStride;
      size_t oStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int O, int I, int H, int W)
      {
        hStride = size_t(W) * wStride;
        iStride = size_t(H) * hStride;
        oStride = size_t(I) * iStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int o, int i, int h, int w) const
      {
        return size_t(o) * oStride +
               size_t(i) * iStride +
               size_t(h) * hStride +
               size_t(w) * wStride;
      }
    };
  };

  template<typename T, int blockSize>
  struct TensorAddressingOIhwBiBo
  {
    static constexpr int B = blockSize;

    static constexpr size_t oStride = sizeof(T);
    static constexpr size_t iStride = B * oStride;
    static constexpr size_t wStride = B * iStride;
    size_t hStride;
    size_t IStride;
    size_t OStride;

    TensorAddressingOIhwBiBo() = default;

    OIDN_HOST_DEVICE_INLINE TensorAddressingOIhwBiBo(int O, int I, int H, int W)
    {
      hStride = size_t(W)   * wStride;
      IStride = size_t(H)   * hStride;
      OStride = size_t(I/B) * IStride;
    }

    OIDN_HOST_DEVICE_INLINE size_t getOffset(int o, int i, int h, int w) const
    {
      return size_t(o/B) * OStride +
             size_t(i/B) * IStride +
             size_t(h)   * hStride +
             size_t(w)   * wStride +
             size_t(i%B) * iStride +
             size_t(o%B) * oStride;
    }
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::OIhw8i8o>
  {
    template<typename T>
    using Addressing = TensorAddressingOIhwBiBo<T, 8>;
  };
  
  template<>
  struct TensorLayoutTraits<TensorLayout::OIhw16i16o>
  {
    template<typename T>
    using Addressing = TensorAddressingOIhwBiBo<T, 16>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::ohwi>
  {
    template<typename T>
    struct Addressing
    {
      static constexpr size_t iStride = sizeof(T);
      size_t wStride;
      size_t hStride;
      size_t oStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int O, int I, int H, int W)
      {
        wStride = size_t(I) * iStride;
        hStride = size_t(W) * wStride;
        oStride = size_t(H) * hStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int o, int i, int h, int w) const
      {
        return size_t(o) * oStride +
               size_t(i) * iStride +
               size_t(h) * hStride +
               size_t(w) * wStride;
      }
    };
  };

  template<typename T, TensorLayout layout>
  using TensorAddressing = typename TensorLayoutTraits<layout>::template Addressing<T>;

} // namespace oidn
