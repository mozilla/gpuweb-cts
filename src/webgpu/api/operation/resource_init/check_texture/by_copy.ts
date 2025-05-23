import { EncodableTextureFormat } from '../../../../format_info.js';
import { virtualMipSize } from '../../../../util/texture/base.js';

import { CheckContents } from './texture_zero_init_test.js';

export const checkContentsByBufferCopy: CheckContents = (
  t,
  params,
  texture,
  state,
  subresourceRange
) => {
  for (const { level: mipLevel, layer } of subresourceRange.each()) {
    const format = params.format as EncodableTextureFormat;

    t.expectSingleColor(texture, format, {
      size: [t.textureWidth, t.textureHeight, t.textureDepth],
      dimension: params.dimension,
      slice: params.dimension === '2d' ? layer : 0,
      layout: { mipLevel, aspect: params.aspect },
      exp: t.stateToTexelComponents[state],
    });
  }
};

export const checkContentsByTextureCopy: CheckContents = (
  t,
  params,
  texture,
  state,
  subresourceRange
) => {
  for (const { level, layer } of subresourceRange.each()) {
    const format = params.format as EncodableTextureFormat;

    const [width, height, depth] = virtualMipSize(
      params.dimension,
      [t.textureWidth, t.textureHeight, t.textureDepth],
      level
    );

    const dst = t.createTextureTracked({
      dimension: params.dimension,
      size: [width, height, depth],
      format: params.format,
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
    });

    const commandEncoder = t.device.createCommandEncoder({ label: 'checkContentsByTextureCopy' });
    commandEncoder.copyTextureToTexture(
      { texture, mipLevel: level, origin: { x: 0, y: 0, z: layer } },
      { texture: dst, mipLevel: 0 },
      { width, height, depthOrArrayLayers: depth }
    );
    t.queue.submit([commandEncoder.finish()]);

    t.expectSingleColor(dst, format, {
      size: [width, height, depth],
      exp: t.stateToTexelComponents[state],
      layout: { mipLevel: 0, aspect: params.aspect },
    });
  }
};
