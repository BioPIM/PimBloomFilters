#ifndef BLOOM_FILTER_DPU_H
#define BLOOM_FILTER_DPU_H

#include <stdint.h>
#include "dpu_utils.h"
#include "../bloom_filters_common.h"

const uint8_t bit_mask [8] = {
    0x01,  //00000001
    0x02,  //00000010
    0x04,  //00000100
    0x08,  //00001000
    0x10,  //00010000
    0x20,  //00100000
    0x40,  //01000000
    0x80   //10000000
};

const uint64_t random_values [256] = {
    0x538575f4081ef4b3,
    0x2a1a2d81200db95f,
    0x74550fef64e6d8c0,
    0x1860da0858251228,
    0x265c279c01a74823,
    0x3c6519531e4b842a,
    0x146ce1b5070acb6b,
    0x79399c6f7bc20e90,
    0x4b188fbb41cb1b68,
    0x6cc63b214c5a9a40,
    0x31368cb70ed8bb27,
    0x4f3592242b740007,
    0x23aee88a11d34fc3,
    0x7bf9a27f206b660c,
    0x2aa2471c41998ed6,
    0x5f503dfd7e27bd11,
    0x49b88389096a6b7f,
    0x1e3576703e0d9379,
    0x6e51443f36965078,
    0x1632a5a114ad6bdb,
    0x383d989b5297bef4,
    0x32f8f0054caa7a51,
    0x59a28a602c328c74,
    0x486c88e124bb1a1b,
    0x6dfda7dd3532c403,
    0x7115b45b1f343494,
    0x440b7f2a404b467f,
    0x4aa8349c67ba67b5,
    0x521e964246a1d71b,
    0x825cdc17cc0dd5f,
    0x83b65f167760bbf,
    0x7ae89a7051f3e97b,
    0x70e0773e191e10e0,
    0x10017cf45f31bb7d,
    0x4fb4615926342295,
    0x73df275907f1f9f4,
    0x78cbe18926d8175e,
    0x549c7445526e6be9,
    0x530aa3d31d08fd27,
    0x7729860441084bb0,
    0x523bc12a683f3a5f,
    0x603c804416474054,
    0x288a80de2ae4b4e0,
    0x7e01a8097aa91721,
    0x71868bfc062775cb,
    0x7769f48079c1f1ed,
    0x6d9d818a72528ef0,
    0x4bb5db685e7df8c8,
    0xb709fd05bb7585c,
    0x3dafb4455b250129,
    0x1eb7af1318edb9e,
    0x6316fb1e7ab75c7b,
    0x5866f2fd37b36f63,
    0x4d25c8642b7196d0,
    0x54bc6c8a444f4e69,
    0x6c79e28026f82db4,
    0x2c8e88c84cb662c4,
    0x3d3f6e09551909a7,
    0x779b17a53b411612,
    0x4fc220c86921a3a1,
    0x41688bdd472c1548,
    0x62e3958e2f060d67,
    0x397ea4382e9970f7,
    0xd84062f44ef4408,
    0xa50c9534b33ba75,
    0x201445320c3c4445,
    0x7cc29613032b4050,
    0x6f3a0c055298910,
    0x3adeafb354196924,
    0x9b1fe00f9b1c3e,
    0x1868b78d6d150260,
    0x369349f244f74056,
    0x39cb652573d2b7fb,
    0x1a1049fd31667cca,
    0x2f13ce0e69d26ac5,
    0x1a88206b707c59eb,
    0x30fe800d7d6bb5f9,
    0x1f8267536a7d2445,
    0x2c0526f02d066d82,
    0x2f6c684d3655f044,
    0x783a27f74f80ad7f,
    0x4292348974fcbe0b,
    0x52abedcf4985d549,
    0x4a26471b0d8a9d83,
    0x1d9f3e6d4ac166fc,
    0x1d25b9c13607f5fb,
    0x37d6695c53b903b3,
    0x7aff365171a1ce81,
    0x478bbbaf150f804e,
    0x23084b4b769f89bd,
    0x7ee1eb133d906bb6,
    0x671be3a82fe06b20,
    0x3afc21b0069e4afb,
    0x1a5d8f65670148a0,
    0x33a4b87e49c9f7b2,
    0x1d5738e42bdee075,
    0x194aa5325fe96d6d,
    0x20db9e806bf69301,
    0x296f42b66b01e59c,
    0x79813084470e8124,
    0x35c34c9816a6ea45,
    0x7d16771f6d99b5f4,
    0x6a5fedf97815ad70,
    0x5f3b847631eba9a8,
    0xd252dbe0243cfc1,
    0x288b33650c0718d1,
    0x3fd43b780fa7170d,
    0x3be783f17ad05d28,
    0x1645620956451356,
    0x61d1a5c849ea1a87,
    0x200f0b087f28dead,
    0x75c8fafc3959b03a,
    0x5f124c1a16a4997d,
    0x2550433c08818ed1,
    0x1a67f191ed173c0,
    0x4f900ff53769cbb1,
    0x35785e064ca68714,
    0x250381a51fd84bff,
    0x44bc3484043f061b,
    0x51c3f5a751e16242,
    0x682d5dd7a4f290c,
    0x5de87b1346571155,
    0x9f6401919cfff04,
    0x41276e7d203ba222,
    0x7015125a22f91445,
    0x6a25bca910241d62,
    0x2221f2f25feeb7a6,
    0x497dcd9d01343f0d,
    0x769351236ece10d9,
    0x9b5cdde7839d03c,
    0xd9f84995945ddd3,
    0x2fa39bed4317e29f,
    0x25ec64e754a71d92,
    0x62f02e9e6aa8996b,
    0x58e623ae34b42445,
    0x3c89fbad5f68f98b,
    0x2f034d511a7276c0,
    0x25c00ae038f98d6b,
    0x344275c466e7795d,
    0x59352f8d2457881e,
    0x9e08da2435aec37,
    0x347ba5802c028095,
    0x2349a3dd7df9731d,
    0x2d36bfa219dcf500,
    0x6cc783f636ec8d80,
    0x1216c53c7a670890,
    0x10326b5341ba6129,
    0x3d7eeb2f361ed03a,
    0x16617ebb206f19ce,
    0x20c769a56f47a269,
    0x55233e135d516552,
    0x4eb09bf404268b65,
    0x77c3dc127470a6d4,
    0x3d2018d02c0651d6,
    0x5b5820311655485d,
    0x505dd9f46538add4,
    0x59b0349404d97f74,
    0x113b2e697cf9d871,
    0x2d2f2923e71ee0b,
    0x16d6cd716f9a7688,
    0x755e7b8b28ed92ad,
    0x6a017f180590e6de,
    0x6aa7f3d627806a48,
    0x3bafb71801097292,
    0x47ef84165c7720bd,
    0x705114fb1d12c229,
    0x39c8860f3f01b0f0,
    0x21394d8e318c6221,
    0x337257c45e59665e,
    0x5d92b3f70eca77f6,
    0x74aeaebc2df08deb,
    0x740325ca4e5ee350,
    0x32ca0d5f053e5433,
    0x4b58bbc2359cfff1,
    0x43b0423e622f8933,
    0x2537767a390ebdc9,
    0xb1d1be10f38f592,
    0x3e9fa4a775c50fb7,
    0x36b95fda7a4f5bbf,
    0x76ce82497ea8e3f0,
    0x56c67c7c671f9745,
    0x1bbba61a108f028b,
    0x262148353cf4f3a8,
    0x421b64ac59939ff9,
    0x1b4e5a071fae18a3,
    0x685e17ef0ffd08c3,
    0x4d9ea68e5c613db9,
    0x5e5bec130068b3ed,
    0x619f91ec29b4a7d5,
    0x3605b3df254fd42a,
    0xbe431095b3d2a59,
    0x5e5e91f317014cea,
    0x6a761feb1cfe369a,
    0xcc65ca1212f7fc6,
    0x174d92590394deeb,
    0x1fd863b66e140ed5,
    0x6ab476303b9409d0,
    0x7ea3116010d5be65,
    0x7888fd7940be760c,
    0x6a695e5e13d75780,
    0x606c8eaf52c7764e,
    0x23d460432e0b353d,
    0x2f28b40702304c56,
    0x2e73e92b10c845f4,
    0x2be4f42c64799d0a,
    0x36181a1e37c92535,
    0x3fb6c7631476ac12,
    0x4eca721f2a2ce74e,
    0x3174e2ac5b90cec0,
    0x4b5c671448c27506,
    0x5f25adab6b34cacb,
    0x36d683db49da23db,
    0x26c8d49b3579953c,
    0x5aafe2401f51d214,
    0x76380b484519409f,
    0x3329299456a499f8,
    0x17e0b6ed56fd89d7,
    0x4afcf3547096af4,
    0x592dd62e3323b860,
    0x57d1b0e80512ca5a,
    0x179d556a0de9cb07,
    0x3cdbef8f57541ccd,
    0x226077190ba661ae,
    0x181041c53d559c5,
    0x6737306e4cdd6b30,
    0x1c97cecb465cde1a,
    0x381235fb536e52a7,
    0x103701f55edb0a97,
    0x8e7e7e36ae6e436,
    0x7e2cdcab7f1ff32b,
    0x300024d531560640,
    0x55c48d2347e0dbc2,
    0x85390175a745c59,
    0xeea46b661816645,
    0xd9814b966bbf79f,
    0x6694309f25356a24,
    0x74a5c2a62370202e,
    0x7c8986f1170639bf,
    0x2f1681dc7e0a8b0d,
    0x6adb9384164db24b,
    0x4ae7f63e07736250,
    0x5caa906502fa2c39,
    0x5ae1b4f76ce1925a,
    0x61d536d063c99cda,
    0x57c876906002137c,
    0x62e9900507c89b65,
    0x115819bc38ae1d29,
    0x4fa9772719aba9d3,
    0x132279825e93bdde,
    0x7b2d101920ba8e3b,
    0x454fb57d61c140b8,
    0x45eff85f39f57823,
    0x53160e742797f51,
    0x50fbb1e23447e2c3,
    0x40840a5e3bd74566,
    0x4a95950e0b6c009c
};

#endif