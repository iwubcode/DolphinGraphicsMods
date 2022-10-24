////////////////////////////// DOLPHIN CONSTANTS NOT EXPOSED TO USER //////////////////////////////////////////////////
#define XE_GTAO_DEPTH_MIP_LEVELS                            5       // this one is hard-coded to 5 for now
#define XE_GTAO_OCCLUSION_TERM_SCALE                    (1.5f)


#if (XE_GTAO_USE_DEFAULT_CONSTANTS != 0)
#define XE_GTAO_DEFAULT_RADIUS_MULTIPLIER               (1.457f  )  // allows us to use different value as compared to ground truth radius to counter inherent screen space biases
#define XE_GTAO_DEFAULT_FALLOFF_RANGE                   (0.615f  )  // distant samples contribute less
#define XE_GTAO_DEFAULT_SAMPLE_DISTRIBUTION_POWER       (2.0f    )  // small crevices more important than big surfaces
#define XE_GTAO_DEFAULT_THIN_OCCLUDER_COMPENSATION      (0.0f    )  // the new 'thickness heuristic' approach
#define XE_GTAO_DEFAULT_FINAL_VALUE_POWER               (2.2f    )  // modifies the final ambient occlusion value using power function - this allows some of the above heuristics to do different things
#define XE_GTAO_DEFAULT_DEPTH_MIP_SAMPLING_OFFSET       (3.30f   )  // main trade-off between performance (memory bandwidth) and quality (temporal stability is the first affected, thin objects next)
#endif

//#define XE_GTAO_GENERATE_NORMALS_INPLACE 1
//#define XE_GTAO_FP32_DEPTHS 1
//#define XE_GTAO_USE_HALF_FLOAT_PRECISION 0

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016-2021, Intel Corporation 
// 
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// XeGTAO is based on GTAO/GTSO "Jimenez et al. / Practical Real-Time Strategies for Accurate Indirect Occlusion", 
// https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
// 
// Implementation:  Filip Strugar (filip.strugar@intel.com), Steve Mccalla <stephen.mccalla@intel.com>         (\_/)
// Version:         (see XeGTAO.h)                                                                            (='.'=)
// Details:         https://github.com/GameTechDev/XeGTAO                                                     (")_(")
//
// Version history: see XeGTAO.h
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef XE_GTAO_SHOW_DEBUG_VIZ
//#include "vaShared.hlsl"
#endif

#if defined( XE_GTAO_SHOW_NORMALS ) || defined( XE_GTAO_SHOW_EDGES ) || defined( XE_GTAO_SHOW_BENT_NORMALS )
//RWTexture2D<float4>         g_outputDbgImage    : register( u2 );
#endif

//#include "XeGTAO.h"

#define XE_GTAO_PI               	(3.1415926535897932384626433832795)
#define XE_GTAO_PI_HALF             (1.5707963267948966192313216916398)

#ifndef XE_GTAO_USE_HALF_FLOAT_PRECISION
#define XE_GTAO_USE_HALF_FLOAT_PRECISION 1
#endif

#if defined(XE_GTAO_FP32_DEPTHS) && XE_GTAO_USE_HALF_FLOAT_PRECISION
#error Using XE_GTAO_USE_HALF_FLOAT_PRECISION with 32bit depths is not supported yet unfortunately (it is possible to apply fp16 on parts not related to depth but this has not been done yet)
#endif 


#if (XE_GTAO_USE_HALF_FLOAT_PRECISION != 0)
	#if 1 // old fp16 approach (<SM6.2)
		#define min16float      float; 
		#define min16float2     float2;
		#define min16float3     float3;
		#define min16float4     float4;
		#define min16float3x3   float3x3;
	#else // new fp16 approach (requires SM6.2 and -enable-16bit-types) - WARNING: perf degradation noticed on some HW, while the old (min16float) path is mostly at least a minor perf gain so this is more useful for quality testing
		#define float16_t       float; 
		#define float16_t2      float2;
		#define float16_t3      float3;
		#define float16_t4      float4;
		#define float16_t3x3    float3x3;
	#endif
#else
	#define float           float;
	#define float2          float2;
	#define float3          float3;
	#define float4          float4;
	#define float3x3        float3x3;
#endif


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// R11G11B10_UNORM <-> float3
/*float3 XeGTAO_R11G11B10_UNORM_to_FLOAT3( uint packedInput )
{
	float3 unpackedOutput;
	unpackedOutput.x = float( ( packedInput       ) & 0x000007ff ) / 2047.0f;
	unpackedOutput.y = float( ( packedInput >> 11 ) & 0x000007ff ) / 2047.0f;
	unpackedOutput.z = float( ( packedInput >> 22 ) & 0x000003ff ) / 1023.0f;
	return unpackedOutput;
}*/
// 'unpackedInput' is float3 and not float3 on purpose as half float lacks precision for below!
/*uint XeGTAO_FLOAT3_to_R11G11B10_UNORM( float3 unpackedInput )
{
	uint packedOutput;
	packedOutput =( ( uint( VA_SATURATE( unpackedInput.x ) * 2047 + 0.5f ) ) |
		( uint( VA_SATURATE( unpackedInput.y ) * 2047 + 0.5f ) << 11 ) |
		( uint( VA_SATURATE( unpackedInput.z ) * 1023 + 0.5f ) << 22 ) );
	return packedOutput;
}*/
//
float4 XeGTAO_R8G8B8A8_UNORM_to_FLOAT4( uint packedInput )
{
	float4 unpackedOutput;
	unpackedOutput.x = float( packedInput & 0x000000ff ) / float(255);
	unpackedOutput.y = float( ( ( packedInput >> 8 ) & 0x000000ff ) ) / float(255);
	unpackedOutput.z = float( ( ( packedInput >> 16 ) & 0x000000ff ) ) / float(255);
	unpackedOutput.w = float( packedInput >> 24 ) / float(255);
	return unpackedOutput;
}
//
uint XeGTAO_FLOAT4_to_R8G8B8A8_UNORM( float4 unpackedInput )
{
	return (( uint( clamp( unpackedInput.x, 0.0, 1.0 ) * float(255) + float(0.5) ) ) |
			( uint( clamp( unpackedInput.y, 0.0, 1.0 ) * float(255) + float(0.5) ) << 8 ) |
			( uint( clamp( unpackedInput.z, 0.0, 1.0 ) * float(255) + float(0.5) ) << 16 ) |
			( uint( clamp( unpackedInput.w, 0.0, 1.0 ) * float(255) + float(0.5) ) << 24 ) );
}

float3 DisplayNormalSRGB( float3 normal )
{
    float x = pow( abs( normal.x * 0.5 + 0.5 ), 2.2 );
    float y = pow( abs( normal.y * 0.5 + 0.5 ), 2.2 );
    float z = pow( abs( normal.z * 0.5 + 0.5 ), 2.2 );
    return float3(x, y, z);
}

float CustomToLinearDepth(float depth)
{
	float NearZ = 0.001f;
	float FarZ = 1.0f;
	const float A = (1.0f - (FarZ / NearZ)) / 2.0f;
	const float B = (1.0f + (FarZ / NearZ)) / 2.0f;
	return 1.0f / (A * depth + B);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float3 XeGTAO_ComputeViewspacePosition( const float2 screenPos, const float viewspaceDepth )
{
	float3 ret;
	ret.xy = (ndc_to_view_mul * screenPos.xy + ndc_to_view_add) * viewspaceDepth;
	ret.z = viewspaceDepth;
	return ret;
}

float XeGTAO_ScreenSpaceToViewSpaceDepth( const float screenDepth )
{
	// Optimised version of "-cameraClipNear / (cameraClipFar - projDepth * (cameraClipFar - cameraClipNear)) * cameraClipFar"
	//return z_depth_linear_mul / (z_depth_linear_add - screenDepth);
	return 0;
}

float4 XeGTAO_CalculateEdges( const float centerZ, const float leftZ, const float rightZ, const float topZ, const float bottomZ )
{
	float4 edgesLRTB = float4( leftZ, rightZ, topZ, bottomZ ) - float(centerZ);

	float slopeLR = (edgesLRTB.y - edgesLRTB.x) * 0.5;
	float slopeTB = (edgesLRTB.w - edgesLRTB.z) * 0.5;
	float4 edgesLRTBSlopeAdjusted = edgesLRTB + float4( slopeLR, -slopeLR, slopeTB, -slopeTB );
	edgesLRTB = min( abs( edgesLRTB ), abs( edgesLRTBSlopeAdjusted ) );
	return float4(clamp( ( 1.25 - edgesLRTB / (centerZ * 0.011) ), 0.0, 1.0 ));
}

// packing/unpacking for edges; 2 bits per edge mean 4 gradient values (0, 0.33, 0.66, 1) for smoother transitions!
float XeGTAO_PackEdges( float4 edgesLRTB )
{
	// integer version:
	// edgesLRTB = clamp(edgesLRTB, 0.0, 1.0) * 2.9.xxxx + 0.5.xxxx;
	// return ((uint(edgesLRTB.x)) << 6) + ((uint(edgesLRTB.y)) << 4) + ((uint(edgesLRTB.z)) << 2) + ((uint(edgesLRTB.w)));
	// 
	// optimized, should be same as above
	edgesLRTB = round( clamp( edgesLRTB, 0.0, 1.0 ) * 2.9 );
	return dot( edgesLRTB, float4( 64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0 ) ) ;
}

float3 XeGTAO_CalculateNormal( const float4 edgesLRTB, float3 pixCenterPos, float3 pixLPos, float3 pixRPos, float3 pixTPos, float3 pixBPos )
{
	// Get this pixel's viewspace normal
	float4 acceptedNormals  = clamp( float4( edgesLRTB.x*edgesLRTB.z, edgesLRTB.z*edgesLRTB.y, edgesLRTB.y*edgesLRTB.w, edgesLRTB.w*edgesLRTB.x ) + 0.01, 0.0, 1.0 );

	pixLPos = normalize(pixLPos - pixCenterPos);
	pixRPos = normalize(pixRPos - pixCenterPos);
	pixTPos = normalize(pixTPos - pixCenterPos);
	pixBPos = normalize(pixBPos - pixCenterPos);

	float3 pixelNormal =  acceptedNormals.x * cross( pixLPos, pixTPos ) +
						+ acceptedNormals.y * cross( pixTPos, pixRPos ) +
						+ acceptedNormals.z * cross( pixRPos, pixBPos ) +
						+ acceptedNormals.w * cross( pixBPos, pixLPos );
	//pixelNormal.y *= -1;
	pixelNormal = normalize( pixelNormal );

	return pixelNormal;
}

#ifdef XE_GTAO_SHOW_DEBUG_VIZ
float4 DbgGetSliceColor(int slice, int sliceCount, bool mirror)
{
	float red = float(slice) / float(sliceCount); float green = 0.01; float blue = 1.0 - float(slice) / float(sliceCount);
	return (mirror)?(float4(blue, green, red, 0.9)):(float4(red, green, blue, 0.9));
}
#endif

// http://h14s.p5r.org/2012/09/0x5f3759df.html, [Drobot2014a] Low Level Optimizations for GCN, https://blog.selfshadow.com/publications/s2016-shading-course/activision/s2016_pbs_activision_occlusion.pdf slide 63
float XeGTAO_FastSqrt( float x )
{
	return float(intBitsToFloat( 0x1fbd1df5 + ( floatBitsToInt( x ) >> 1 ) ));
}
// input [-1, 1] and output [0, PI], from https://seblagarde.wordpress.com/2014/12/01/inverse-trigonometric-functions-gpu-optimization-for-amd-gcn-architecture/
float XeGTAO_FastACos( float inX )
{ 
	const float PI = 3.141593;
	const float HALF_PI = 1.570796;
	float x = abs(inX); 
	float res = -0.156583 * x + HALF_PI; 
	res *= XeGTAO_FastSqrt(1.0 - x); 
	return (inX >= 0) ? res : PI - res; 
}

float4 XeGTAO_EncodeVisibilityBentNormal( float visibility, float3 bentNormal )
{
	return float4( bentNormal * 0.5 + 0.5, visibility );
	//return XeGTAO_FLOAT4_to_R8G8B8A8_UNORM( float4( bentNormal * 0.5 + 0.5, visibility ) );
}

void XeGTAO_DecodeVisibilityBentNormal( const uint packedValue, out float visibility, out float3 bentNormal )
{
	float4 decoded = XeGTAO_R8G8B8A8_UNORM_to_FLOAT4( packedValue );
	bentNormal = decoded.xyz * 2.0.xxx - 1.0.xxx;   // could normalize - don't want to since it's done so many times, better to do it at the final step only
	visibility = decoded.w;
}

void XeGTAO_OutputWorkingTerm( const uint2 pixCoord, float visibility, float3 bentNormal )
{
	visibility = clamp( visibility / float(XE_GTAO_OCCLUSION_TERM_SCALE), 0.0, 1.0 );
#if XE_GTAO_COMPUTE_BENT_NORMALS == 1
	float4 f = XeGTAO_EncodeVisibilityBentNormal( visibility, bentNormal );
	SetOutput(f, int3(pixCoord, 0));
#else
	//float f = visibility * 255.0 + 0.5;
	//SetOutput(float4(f, f, f, 1.0), int3(pixCoord, 0));
	SetOutput(float4(visibility, visibility, visibility, 1.0), int3(pixCoord, 0));
#endif
}

// "Efficiently building a matrix to rotate one vector to another"
// http://cs.brown.edu/research/pubs/pdfs/1999/Moller-1999-EBA.pdf / https://dl.acm.org/doi/10.1080/10867651.1999.10487509
// (using https://github.com/assimp/assimp/blob/master/include/assimp/matrix3x3.inl#L275 as a code reference as it seems to be best)
float3x3 XeGTAO_RotFromToMatrix( float3 from, float3 to )
{
	const float e       = dot(from, to);
	const float f       = abs(e); //(e < 0)? -e:e;

	// WARNING: This has not been tested/worked through, especially not for 16bit floats; seems to work in our special use case (from is always {0, 0, -1}) but wouldn't use it in general
	if( f > float( 1.0 - 0.0003 ) )
		return float3x3( 1, 0, 0, 0, 1, 0, 0, 0, 1 );

	const float3 v      = cross( from, to );
	/* ... use this hand optimized version (9 mults less) */
	const float h       = (1.0)/(1.0 + e);      /* optimization by Gottfried Chen */
	const float hvx     = h * v.x;
	const float hvz     = h * v.z;
	const float hvxy    = hvx * v.y;
	const float hvxz    = hvx * v.z;
	const float hvyz    = hvz * v.y;

	float3x3 mtx;
	mtx[0][0] = e + hvx * v.x;
	mtx[0][1] = hvxy - v.z;
	mtx[0][2] = hvxz + v.y;

	mtx[1][0] = hvxy + v.z;
	mtx[1][1] = e + h * v.y * v.y;
	mtx[1][2] = hvyz - v.x;

	mtx[2][0] = hvxz - v.y;
	mtx[2][1] = hvyz + v.x;
	mtx[2][2] = e + hvz * v.z;

	return mtx;
}

float ConvertDepth(float depth)
{
	//return XeGTAO_ScreenSpaceToViewSpaceDepth(DEPTH_VALUE(depth));
	return CustomToLinearDepth(DEPTH_VALUE(depth));
	//return depth;
}

void XeGTAO_MainPass( const uint2 pixCoord, float sliceCount, float stepsPerSlice, const float2 localNoise, float3 viewspaceNormal )
{                                                                       
	float2 normalizedScreenPos = (pixCoord + 0.5.xx) * GetInvResolution();
	
	float4 valuesUL   = gather_red(DEPTH_BUFFER_INPUT_INDEX, float3( float2(pixCoord) * GetInvResolution(), 0), int2(0, 0) );
	float4 valuesBR   = gather_red(DEPTH_BUFFER_INPUT_INDEX, float3( float2(pixCoord) * GetInvResolution(), 0), int2(1, 1) );
	
	// viewspace Z at the center
	float viewspaceZ  = ConvertDepth(valuesUL.y);

	// viewspace Zs left top right bottom
	const float pixLZ = ConvertDepth(valuesUL.x);
	const float pixTZ = ConvertDepth(valuesUL.z);
	const float pixRZ = ConvertDepth(valuesBR.z);
	const float pixBZ = ConvertDepth(valuesBR.x);

	float4 edgesLRTB  = XeGTAO_CalculateEdges( float(viewspaceZ), float(pixLZ), float(pixRZ), float(pixTZ), float(pixBZ) );
	SetAdditionalOutput(WORKINGEDGES_OUTPUT_INDEX, edgesLRTB, int3(pixCoord, 0) );

	// Generating screen space normals in-place is faster than generating normals in a separate pass but requires
	// use of 32bit depth buffer (16bit works but visibly degrades quality) which in turn slows everything down. So to
	// reduce complexity and allow for screen space normal reuse by other effects, we've pulled it out into a separate
	// pass.
	// However, we leave this code in, in case anyone has a use-case where it fits better.
#ifdef XE_GTAO_GENERATE_NORMALS_INPLACE
	float3 CENTER   = XeGTAO_ComputeViewspacePosition( normalizedScreenPos, viewspaceZ );
	float3 LEFT     = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + float2(-1,  0) * GetInvResolution(), pixLZ );
	float3 RIGHT    = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + float2( 1,  0) * GetInvResolution(), pixRZ );
	float3 TOP      = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + float2( 0, -1) * GetInvResolution(), pixTZ );
	float3 BOTTOM   = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + float2( 0,  1) * GetInvResolution(), pixBZ );
	viewspaceNormal = float3(XeGTAO_CalculateNormal( edgesLRTB, CENTER, LEFT, RIGHT, TOP, BOTTOM ));
#endif

	// Move center pixel slightly towards camera to avoid imprecision artifacts due to depth buffer imprecision; offset depends on depth texture format used
#ifdef XE_GTAO_FP32_DEPTHS
	viewspaceZ *= 0.99999;     // this is good for FP32 depth buffer
#else
	viewspaceZ *= 0.99920;     // this is good for FP16 depth buffer
#endif

	const float3 pixCenterPos   = XeGTAO_ComputeViewspacePosition( normalizedScreenPos, viewspaceZ );
	const float3 viewVec      = float3(normalize(-pixCenterPos));
	
	// prevents normals that are facing away from the view vector - xeGTAO struggles with extreme cases, but in Vanilla it seems rare so it's disabled by default
	// viewspaceNormal = normalize( viewspaceNormal + max( 0, -dot( viewspaceNormal, viewVec ) ) * viewVec );

#if XE_GTAO_SHOW_NORMALS == 1
	SetAdditionalOutput(DEBUG_IMG_OUTPUT_INDEX, float4( DisplayNormalSRGB( viewspaceNormal.xyz ), 1 ), int3(pixCoord, 0));
#endif

#if XE_GTAO_SHOW_EDGES == 1
	SetAdditionalOutput(DEBUG_IMG_OUTPUT_INDEX, 1.0 - float4( edgesLRTB.x, edgesLRTB.y * 0.5 + edgesLRTB.w * 0.5, edgesLRTB.z, 1.0 ), int3(pixCoord, 0));
#endif

#if XE_GTAO_USE_DEFAULT_CONSTANTS != 0
	const float effectRadius              = float(GetOption(XE_GTAO_EFFECT_RADIUS)) * float(XE_GTAO_DEFAULT_RADIUS_MULTIPLIER);
	const float sampleDistributionPower   = float(XE_GTAO_DEFAULT_SAMPLE_DISTRIBUTION_POWER);
	const float thinOccluderCompensation  = float(XE_GTAO_DEFAULT_THIN_OCCLUDER_COMPENSATION);
	const float falloffRange              = float(XE_GTAO_DEFAULT_FALLOFF_RANGE) * effectRadius;
#else
	const float effectRadius              = float(GetOption(XE_GTAO_EFFECT_RADIUS)) * float(GetOption(XE_GTAO_RADIUS_MULTIPLIER));
	const float sampleDistributionPower   = float(GetOption(XE_GTAO_SAMPLE_DISTRIBUTION_POWER));
	const float thinOccluderCompensation  = float(GetOption(XE_GTAO_THIN_OCCLUDER_COMPENSATION));
	const float falloffRange              = float(GetOption(XE_GTAO_FALLOFF_RANGE)) * float(effectRadius);
#endif

	const float falloffFrom       = effectRadius * (float(1)-float(GetOption(XE_GTAO_FALLOFF_RANGE)));

	// fadeout precompute optimisation
	const float falloffMul        = float(-1.0) / ( falloffRange );
	const float falloffAdd        = falloffFrom / ( falloffRange ) + float(1.0);

	float visibility = 0;
#if XE_GTAO_COMPUTE_BENT_NORMALS == 1
	float3 bentNormal = float3(0, 0, 0);
#else
	float3 bentNormal = viewspaceNormal;
#endif

#ifdef XE_GTAO_SHOW_DEBUG_VIZ
	float3 dbgWorldPos          = mul(g_globals.ViewInv, float4(pixCenterPos, 1)).xyz;
#endif

	// see "Algorithm 1" in https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
	{
		const float noiseSlice  = float(localNoise.x);
		const float noiseSample = float(localNoise.y);

		// quality settings / tweaks / hacks
		const float pixelTooCloseThreshold  = 1.3;      // if the offset is under approx pixel size (pixelTooCloseThreshold), push it out to the minimum distance

		// approx viewspace pixel size at pixCoord; approximation of NDCToViewspace( normalizedScreenPos.xy + pixelsize, pixCenterPos.z ).xy - pixCenterPos.xy;
		const float2 pixelDirRBViewspaceSizeAtCenterZ = viewspaceZ.xx * ndc_to_view_mul * GetInvResolution();

		float screenspaceRadius   = effectRadius / float(pixelDirRBViewspaceSizeAtCenterZ.x);

		// fade out for small screen radii 
		visibility += clamp((10 - screenspaceRadius)/100, 0.0, 1.0)*0.5;

#if 0   // sensible early-out for even more performance; disabled because not yet tested
		[branch]
		if( screenspaceRadius < pixelTooCloseThreshold )
		{
			XeGTAO_OutputWorkingTerm( pixCoord, 1, viewspaceNormal );
			return;
		}
#endif

#ifdef XE_GTAO_SHOW_DEBUG_VIZ
		[branch] if (IsUnderCursorRange(pixCoord, int2(1, 1)))
		{
			float3 dbgWorldNorm     = mul((float3x3)g_globals.ViewInv, viewspaceNormal).xyz;
			float3 dbgWorldViewVec  = mul((float3x3)g_globals.ViewInv, viewVec).xyz;
			//DebugDraw3DArrow(dbgWorldPos, dbgWorldPos + 0.5 * dbgWorldViewVec, 0.02, float4(0, 1, 0, 0.95));
			//DebugDraw2DCircle(pixCoord, screenspaceRadius, float4(1, 0, 0.2, 1));
			DebugDraw3DSphere(dbgWorldPos, effectRadius, float4(1, 0.2, 0, 0.1));
			//DebugDraw3DText(dbgWorldPos, float2(0, 0), float4(0.6, 0.3, 0.3, 1), float4( pixelDirRBViewspaceSizeAtCenterZ.xy, 0, screenspaceRadius) );
		}
#endif

		// this is the min distance to start sampling from to avoid sampling from the center pixel (no useful data obtained from sampling center pixel)
		const float minS = float(pixelTooCloseThreshold) / screenspaceRadius;

		for( float slice = 0; slice < sliceCount; slice++ )
		{
			float sliceK = (slice+noiseSlice) / sliceCount;
			// lines 5, 6 from the paper
			float phi = sliceK * XE_GTAO_PI;
			float cosPhi = cos(phi);
			float sinPhi = sin(phi);
			float2 omega = float2(cosPhi, -sinPhi);       //float2 on omega causes issues with big radii

			// convert to screen units (pixels) for later use
			omega *= screenspaceRadius;

			// line 8 from the paper
			const float3 directionVec = float3(cosPhi, sinPhi, 0);

			// line 9 from the paper
			const float3 orthoDirectionVec = directionVec - (dot(directionVec, viewVec) * viewVec);

			// line 10 from the paper
			//axisVec is orthogonal to directionVec and viewVec, used to define projectedNormal
			const float3 axisVec = normalize( cross(orthoDirectionVec, viewVec) );

			// alternative line 9 from the paper
			// float3 orthoDirectionVec = cross( viewVec, axisVec );

			// line 11 from the paper
			float3 projectedNormalVec = viewspaceNormal - axisVec * dot(viewspaceNormal, axisVec);

			// line 13 from the paper
			float signNorm = float(sign( dot( orthoDirectionVec, projectedNormalVec ) ));

			// line 14 from the paper
			float projectedNormalVecLength = length(projectedNormalVec);
			float cosNorm = float(clamp(dot(projectedNormalVec, viewVec) / projectedNormalVecLength, 0.0, 1.0));

			// line 15 from the paper
			float n = signNorm * XeGTAO_FastACos(cosNorm);

			// this is a lower weight target; not using -1 as in the original paper because it is under horizon, so a 'weight' has different meaning based on the normal
			const float lowHorizonCos0  = cos(n+XE_GTAO_PI_HALF);
			const float lowHorizonCos1  = cos(n-XE_GTAO_PI_HALF);

			// lines 17, 18 from the paper, manually unrolled the 'side' loop
			float horizonCos0           = lowHorizonCos0; //-1;
			float horizonCos1           = lowHorizonCos1; //-1;

			for( float step = 0; step < stepsPerSlice; step++ )
			{
				// R1 sequence (http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/)
				const float stepBaseNoise = float(slice + step * stepsPerSlice) * 0.6180339887498948482; // <- this should unroll
				float stepNoise = fract(noiseSample + stepBaseNoise);

				// approx line 20 from the paper, with added noise
				float s = (step+stepNoise) / (stepsPerSlice); // + float2(1e-6f));

				// additional distribution modifier
				s       = float(pow( s, float(sampleDistributionPower) ));

				// avoid sampling center pixel
				s       += minS;

				// approx lines 21-22 from the paper, unrolled
				float2 sampleOffset = s * omega;

				float sampleOffsetLength = length( sampleOffset );

				// note: when sampling, using point_point_point or point_point_linear sampler works, but linear_linear_linear will cause unwanted interpolation between neighbouring depth values on the same MIP level!
				const float mipLevel    = float(clamp( log2( sampleOffsetLength ) - GetOption(XE_GTAO_DEPTH_MIP_SAMPLING_OFFSET), 0, XE_GTAO_DEPTH_MIP_LEVELS ));

				// Snap to pixel center (more correct direction math, avoids artifacts due to sampling pos not matching depth texel center - messes up slope - but adds other 
				// artifacts due to them being pushed off the slice). Also use full precision for high res cases.
				sampleOffset = round(sampleOffset) * float2(GetInvResolution());

#ifdef XE_GTAO_SHOW_DEBUG_VIZ
				int mipLevelU = (int)round(mipLevel);
				float4 mipColor = clamp( float4( mipLevelU>=3, mipLevelU>=1 && mipLevelU<=3, mipLevelU<=1, 1.0 ), 0.0, 1.0 );
				if( all( sampleOffset == 0 ) )
					DebugDraw2DText( pixCoord, float4( 1, 0, 0, 1), pixelTooCloseThreshold );
				[branch] if (IsUnderCursorRange(pixCoord, int2(1, 1)))
				{
					//DebugDraw2DText( (normalizedScreenPos + sampleOffset) * GetResolution(), mipColor, mipLevelU );
					//DebugDraw2DText( (normalizedScreenPos + sampleOffset) * GetResolution(), mipColor, (uint)slice );
					//DebugDraw2DText( (normalizedScreenPos - sampleOffset) * GetResolution(), mipColor, (uint)slice );
					//DebugDraw2DText( (normalizedScreenPos - sampleOffset) * GetResolution(), clamp( float4( mipLevelU>=3, mipLevelU>=1 && mipLevelU<=3, mipLevelU<=1, 1.0 ), 0.0, 1.0 ), mipLevelU );
				}
#endif

				float2 sampleScreenPos0 = normalizedScreenPos + sampleOffset;
				float SZ0 = ConvertDepth(SampleInputLodLocation(DEPTH_BUFFER_INPUT_INDEX, mipLevel, sampleScreenPos0).x);
				float3 samplePos0 = XeGTAO_ComputeViewspacePosition( sampleScreenPos0, SZ0 );

				float2 sampleScreenPos1 = normalizedScreenPos - sampleOffset;
				float SZ1 = ConvertDepth(SampleInputLodLocation(DEPTH_BUFFER_INPUT_INDEX, mipLevel, sampleScreenPos1).x);
				float3 samplePos1 = XeGTAO_ComputeViewspacePosition( sampleScreenPos1, SZ1 );

				float3 sampleDelta0     = (samplePos0 - float3(pixCenterPos)); // using float for sampleDelta causes precision issues
				float3 sampleDelta1     = (samplePos1 - float3(pixCenterPos)); // using float for sampleDelta causes precision issues
				float sampleDist0     = float(length( sampleDelta0 ));
				float sampleDist1     = float(length( sampleDelta1 ));

				// approx lines 23, 24 from the paper, unrolled
				float3 sampleHorizonVec0 = float3((sampleDelta0 / sampleDist0));
				float3 sampleHorizonVec1 = float3((sampleDelta1 / sampleDist1));

				// any sample out of radius should be discarded - also use fallof range for smooth transitions; this is a modified idea from "4.3 Implementation details, Bounding the sampling area"
#if XE_GTAO_USE_DEFAULT_CONSTANTS != 0 && XE_GTAO_DEFAULT_THIN_OBJECT_HEURISTIC == 0
				float weight0         = clamp( sampleDist0 * falloffMul + falloffAdd, 0.0, 1.0 );
				float weight1         = clamp( sampleDist1 * falloffMul + falloffAdd, 0.0, 1.0 );
#else
				// this is our own thickness heuristic that relies on sooner discarding samples behind the center
				float falloffBase0    = length( float3(sampleDelta0.x, sampleDelta0.y, sampleDelta0.z * (1+thinOccluderCompensation) ) );
				float falloffBase1    = length( float3(sampleDelta1.x, sampleDelta1.y, sampleDelta1.z * (1+thinOccluderCompensation) ) );
				float weight0         = clamp( falloffBase0 * falloffMul + falloffAdd, 0.0, 1.0 );
				float weight1         = clamp( falloffBase1 * falloffMul + falloffAdd, 0.0, 1.0 );
#endif

				// sample horizon cos
				float shc0 = float(dot(sampleHorizonVec0, viewVec));
				float shc1 = float(dot(sampleHorizonVec1, viewVec));

				// discard unwanted samples
				shc0 = mix( lowHorizonCos0, shc0, weight0 ); // this would be more correct but too expensive: cos(mix( acos(lowHorizonCos0), acos(shc0), weight0 ));
				shc1 = mix( lowHorizonCos1, shc1, weight1 ); // this would be more correct but too expensive: cos(mix( acos(lowHorizonCos1), acos(shc1), weight1 ));

				// thickness heuristic - see "4.3 Implementation details, Height-field assumption considerations"
#if 0   // (disabled, not used) this should match the paper
				float newhorizonCos0 = max( horizonCos0, shc0 );
				float newhorizonCos1 = max( horizonCos1, shc1 );
				horizonCos0 = (horizonCos0 > shc0)?( mix( newhorizonCos0, shc0, thinOccluderCompensation ) ):( newhorizonCos0 );
				horizonCos1 = (horizonCos1 > shc1)?( mix( newhorizonCos1, shc1, thinOccluderCompensation ) ):( newhorizonCos1 );
#elif 0 // (disabled, not used) this is slightly different from the paper but cheaper and provides very similar results
				horizonCos0 = mix( max( horizonCos0, shc0 ), shc0, thinOccluderCompensation );
				horizonCos1 = mix( max( horizonCos1, shc1 ), shc1, thinOccluderCompensation );
#else   // this is a version where thicknessHeuristic is completely disabled
				horizonCos0 = max( horizonCos0, shc0 );
				horizonCos1 = max( horizonCos1, shc1 );
#endif


#ifdef XE_GTAO_SHOW_DEBUG_VIZ
				[branch] if (IsUnderCursorRange(pixCoord, int2(1, 1)))
				{
					float3 WS_samplePos0 = mul(g_globals.ViewInv, float4(samplePos0, 1)).xyz;
					float3 WS_samplePos1 = mul(g_globals.ViewInv, float4(samplePos1, 1)).xyz;
					float3 WS_sampleHorizonVec0 = mul( (float3x3)g_globals.ViewInv, sampleHorizonVec0).xyz;
					float3 WS_sampleHorizonVec1 = mul( (float3x3)g_globals.ViewInv, sampleHorizonVec1).xyz;
					// DebugDraw3DSphere( WS_samplePos0, GetOption(XE_GTAO_EFFECT_RADIUS) * 0.02, DbgGetSliceColor(slice, sliceCount, false) );
					// DebugDraw3DSphere( WS_samplePos1, GetOption(XE_GTAO_EFFECT_RADIUS) * 0.02, DbgGetSliceColor(slice, sliceCount, true) );
					DebugDraw3DSphere( WS_samplePos0, GetOption(XE_GTAO_EFFECT_RADIUS) * 0.02, mipColor );
					DebugDraw3DSphere( WS_samplePos1, GetOption(XE_GTAO_EFFECT_RADIUS) * 0.02, mipColor );
					// DebugDraw3DArrow( WS_samplePos0, WS_samplePos0 - WS_sampleHorizonVec0, 0.002, float4(1, 0, 0, 1 ) );
					// DebugDraw3DArrow( WS_samplePos1, WS_samplePos1 - WS_sampleHorizonVec1, 0.002, float4(1, 0, 0, 1 ) );
					// DebugDraw3DText( WS_samplePos0, float2(0,  0), float4( 1, 0, 0, 1), weight0 );
					// DebugDraw3DText( WS_samplePos1, float2(0,  0), float4( 1, 0, 0, 1), weight1 );

					// DebugDraw2DText( float2( 500, 94+(step+slice*3)*12 ), float4( 0, 1, 0, 1 ), float4( projectedNormalVecLength, 0, horizonCos0, horizonCos1 ) );
				}
#endif
			}

#if 1       // I can't figure out the slight overdarkening on high slopes, so I'm adding this fudge - in the training set, 0.05 is close (PSNR 21.34) to disabled (PSNR 21.45)
			projectedNormalVecLength = mix( projectedNormalVecLength, 1, 0.05 );
#endif

			// line ~27, unrolled
			float h0 = -XeGTAO_FastACos(float(horizonCos1));
			float h1 = XeGTAO_FastACos(float(horizonCos0));
#if 0       // we can skip clamping for a tiny little bit more performance
			h0 = n + clamp( h0-n, float(-XE_GTAO_PI_HALF), float(XE_GTAO_PI_HALF) );
			h1 = n + clamp( h1-n, float(-XE_GTAO_PI_HALF), float(XE_GTAO_PI_HALF) );
#endif
			float iarc0 = (float(cosNorm) + float(2) * float(h0) * float(sin(n))-float(cos(float(2) * float(h0)-n)))/float(4);
			float iarc1 = (float(cosNorm) + float(2) * float(h1) * float(sin(n))-float(cos(float(2) * float(h1)-n)))/float(4);
			float localVisibility = float(projectedNormalVecLength) * float(iarc0+iarc1);
			visibility += localVisibility;

#if XE_GTAO_COMPUTE_BENT_NORMALS == 1
			// see "Algorithm 2 Extension that computes bent normals b."
			float t0 = (6*sin(h0-n)-sin(3*h0-n)+6*sin(h1-n)-sin(3*h1-n)+16*sin(n)-3*(sin(h0+n)+sin(h1+n)))/12;
			float t1 = (-cos(3 * h0-n)-cos(3 * h1-n) +8 * cos(n)-3 * (cos(h0+n) +cos(h1+n)))/12;
			float3 localBentNormal = float3( directionVec.x * float(t0), directionVec.y * float(t0), - float(t1) );
			localBentNormal = float3(mul( XeGTAO_RotFromToMatrix( float3(0,0,-1), viewVec ), localBentNormal )) * projectedNormalVecLength;
			bentNormal += localBentNormal;
#endif
		}
		visibility /= float(sliceCount);
		visibility = pow( visibility, float(GetOption(XE_GTAO_FINAL_VALUE_POWER)) );
		visibility = max( float(0.03), visibility ); // disallow total occlusion (which wouldn't make any sense anyhow since pixel is visible but also helps with packing bent normals)

#if XE_GTAO_COMPUTE_BENT_NORMALS == 1
		bentNormal = normalize(bentNormal) ;
#endif
	}

#if defined(XE_GTAO_SHOW_DEBUG_VIZ) && defined(XE_GTAO_COMPUTE_BENT_NORMALS)
	[branch] if (IsUnderCursorRange(pixCoord, int2(1, 1)))
	{
		float3 dbgWorldViewNorm = mul((float3x3)g_globals.ViewInv, viewspaceNormal).xyz;
		float3 dbgWorldBentNorm = mul((float3x3)g_globals.ViewInv, bentNormal).xyz;
		DebugDraw3DSphereCone( dbgWorldPos, dbgWorldViewNorm, 0.3, VA_PI*0.5 - acos(clamp(visibility, 0.0, 1.0)), float4( 0.2, 0.2, 0.2, 0.5 ) );
		DebugDraw3DSphereCone( dbgWorldPos, dbgWorldBentNorm, 0.3, VA_PI*0.5 - acos(clamp(visibility, 0.0, 1.0)), float4( 0.0, 1.0, 0.0, 0.7 ) );
	}
#endif

	XeGTAO_OutputWorkingTerm( pixCoord, visibility, bentNormal );
}

// weighted average depth filter
/*float XeGTAO_DepthMIPFilter( float depth0, float depth1, float depth2, float depth3, const GTAOConstants consts )
{
	float maxDepth = max( max( depth0, depth1 ), max( depth2, depth3 ) );

	const float depthRangeScaleFactor = 0.75; // found empirically :)
#if XE_GTAO_USE_DEFAULT_CONSTANTS != 0
	const float effectRadius              = depthRangeScaleFactor * float(GetOption(XE_GTAO_EFFECT_RADIUS)) * float(GetOption(XE_GTAO_DEFAULT_RADIUS_MULTIPLIER));
	const float falloffRange              = float(XE_GTAO_DEFAULT_FALLOFF_RANGE) * effectRadius;
#else
	const float effectRadius              = depthRangeScaleFactor * float(GetOption(XE_GTAO_EFFECT_RADIUS)) * float(GetOption(XE_GTAO_RADIUS_MULTIPLIER));
	const float falloffRange              = float(GetOption(XE_GTAO_FALLOFF_RANGE)) * effectRadius;
#endif
	const float falloffFrom       = effectRadius * (float(1)-float(GetOption(XE_GTAO_FALLOFF_RANGE)));
	// fadeout precompute optimisation
	const float falloffMul        = float(-1.0) / ( falloffRange );
	const float falloffAdd        = falloffFrom / ( falloffRange ) + float(1.0);

	float weight0 = clamp( (maxDepth-depth0) * falloffMul + falloffAdd, 0.0, 1.0 );
	float weight1 = clamp( (maxDepth-depth1) * falloffMul + falloffAdd, 0.0, 1.0 );
	float weight2 = clamp( (maxDepth-depth2) * falloffMul + falloffAdd, 0.0, 1.0 );
	float weight3 = clamp( (maxDepth-depth3) * falloffMul + falloffAdd, 0.0, 1.0 );

	float weightSum = weight0 + weight1 + weight2 + weight3;
	return (weight0 * depth0 + weight1 * depth1 + weight2 * depth2 + weight3 * depth3) / weightSum;
}

// This is also a good place to do non-linear depth conversion for cases where one wants the 'radius' (effectively the threshold between near-field and far-field GI), 
// is required to be non-linear (i.e. very large outdoors environments).
float XeGTAO_ClampDepth( float depth )
{
#ifdef XE_GTAO_USE_HALF_FLOAT_PRECISION
	return float(clamp( depth, 0.0, 65504.0 ));
#else
	return clamp( depth, 0.0, 3.402823466e+38 );
#endif
}

void XeGTAO_PrefilterDepths16x16()
{
	// MIP 0
	const uint2 baseCoord = dispatchThreadID;
	const uint2 pixCoord = baseCoord * 2;
	float4 depths4 = sourceNDCDepth.GatherRed( depthSampler, float2( pixCoord * GetInvResolution() ), int2(1,1) );
	float depth0 = XeGTAO_ClampDepth( XeGTAO_ScreenSpaceToViewSpaceDepth( depths4.w, consts ) );
	float depth1 = XeGTAO_ClampDepth( XeGTAO_ScreenSpaceToViewSpaceDepth( depths4.z, consts ) );
	float depth2 = XeGTAO_ClampDepth( XeGTAO_ScreenSpaceToViewSpaceDepth( depths4.x, consts ) );
	float depth3 = XeGTAO_ClampDepth( XeGTAO_ScreenSpaceToViewSpaceDepth( depths4.y, consts ) );
	outDepth0[ pixCoord + uint2(0, 0) ] = float(depth0);
	outDepth0[ pixCoord + uint2(1, 0) ] = float(depth1);
	outDepth0[ pixCoord + uint2(0, 1) ] = float(depth2);
	outDepth0[ pixCoord + uint2(1, 1) ] = float(depth3);

	// MIP 1
	float dm1 = XeGTAO_DepthMIPFilter( depth0, depth1, depth2, depth3, consts );
	outDepth1[ baseCoord ] = float(dm1);
	g_scratchDepths[ groupThreadID.x ][ groupThreadID.y ] = dm1;

	GroupMemoryBarrierWithGroupSync( );

	// MIP 2
	[branch]
	if( all( ( groupThreadID.xy % 2.xx ) == 0 ) )
	{
		float inTL = g_scratchDepths[groupThreadID.x+0][groupThreadID.y+0];
		float inTR = g_scratchDepths[groupThreadID.x+1][groupThreadID.y+0];
		float inBL = g_scratchDepths[groupThreadID.x+0][groupThreadID.y+1];
		float inBR = g_scratchDepths[groupThreadID.x+1][groupThreadID.y+1];

		float dm2 = XeGTAO_DepthMIPFilter( inTL, inTR, inBL, inBR, consts );
		outDepth2[ baseCoord / 2 ] = float(dm2);
		g_scratchDepths[ groupThreadID.x ][ groupThreadID.y ] = dm2;
	}

	GroupMemoryBarrierWithGroupSync( );

	// MIP 3
	[branch]
	if( all( ( groupThreadID.xy % 4.xx ) == 0 ) )
	{
		float inTL = g_scratchDepths[groupThreadID.x+0][groupThreadID.y+0];
		float inTR = g_scratchDepths[groupThreadID.x+2][groupThreadID.y+0];
		float inBL = g_scratchDepths[groupThreadID.x+0][groupThreadID.y+2];
		float inBR = g_scratchDepths[groupThreadID.x+2][groupThreadID.y+2];

		float dm3 = XeGTAO_DepthMIPFilter( inTL, inTR, inBL, inBR, consts );
		outDepth3[ baseCoord / 4 ] = float(dm3);
		g_scratchDepths[ groupThreadID.x ][ groupThreadID.y ] = dm3;
	}

	GroupMemoryBarrierWithGroupSync( );

	// MIP 4
	[branch]
	if( all( ( groupThreadID.xy % 8.xx ) == 0 ) )
	{
		float inTL = g_scratchDepths[groupThreadID.x+0][groupThreadID.y+0];
		float inTR = g_scratchDepths[groupThreadID.x+4][groupThreadID.y+0];
		float inBL = g_scratchDepths[groupThreadID.x+0][groupThreadID.y+4];
		float inBR = g_scratchDepths[groupThreadID.x+4][groupThreadID.y+4];

		float dm4 = XeGTAO_DepthMIPFilter( inTL, inTR, inBL, inBR, consts );
		outDepth4[ baseCoord / 8 ] = float(dm4);
		//g_scratchDepths[ groupThreadID.x ][ groupThreadID.y ] = dm4;
	}
}*/

float4 XeGTAO_UnpackEdges( float _packedVal )
{
	uint packedVal = uint(_packedVal * 255.5);
	float4 edgesLRTB;
	edgesLRTB.x = float((packedVal >> 6) & 0x03) / 3.0;          // there's really no need for mask (as it's an 8 bit input) but I'll leave it in so it doesn't cause any trouble in the future
	edgesLRTB.y = float((packedVal >> 4) & 0x03) / 3.0;
	edgesLRTB.z = float((packedVal >> 2) & 0x03) / 3.0;
	edgesLRTB.w = float((packedVal >> 0) & 0x03) / 3.0;

	return clamp( edgesLRTB, 0.0, 1.0 );
}

#if XE_GTAO_COMPUTE_BENT_NORMALS == 1
#define AOTermType float4            // .xyz is bent normal, .w is visibility term
#else
#define AOTermType float             // .x is visibility term
#endif

void XeGTAO_AddSample( AOTermType ssaoValue, float edgeValue, inout AOTermType sum, inout float sumWeight )
{
	float weight = edgeValue;    

	sum += (weight * ssaoValue);
	sumWeight += weight;
}

void XeGTAO_Output( uint2 pixCoord, AOTermType outputValue, const bool finalApply )
{
#if XE_GTAO_COMPUTE_BENT_NORMALS == 1
	float     visibility = outputValue.w * ((finalApply)?(float(XE_GTAO_OCCLUSION_TERM_SCALE)):(1));
	float3    bentNormal = normalize(outputValue.xyz);
	float f = float(XeGTAO_EncodeVisibilityBentNormal( visibility, bentNormal ));
	SetOutput(float4(f, f, f, 1.0), int3(pixCoord, 0));
	//SetOutput(XeGTAO_EncodeVisibilityBentNormal( visibility, bentNormal ), int3(pixCoord, 0));
#else
	outputValue *=  (finalApply)?(float(XE_GTAO_OCCLUSION_TERM_SCALE)):(1);
	//float f = outputValue * 255.0 + 0.5;
	//SetOutput(float4(f, f, f, 1.0), int3(pixCoord, 0));
	SetOutput(float4(outputValue, outputValue, outputValue, 1.0), int3(pixCoord, 0));
#endif
}

void XeGTAO_DecodeGatherPartial( const float4 packedValue, out AOTermType outDecoded[4] )
{
	for( int i = 0; i < 4; i++ )
#if XE_GTAO_COMPUTE_BENT_NORMALS == 1
		XeGTAO_DecodeVisibilityBentNormal( packedValue[i], outDecoded[i].w, outDecoded[i].xyz );
#else
		//outDecoded[i] = float(packedValue[i]) / float(255.0);
		outDecoded[i] = float(packedValue[i]);
#endif
}

void XeGTAO_Denoise( const uint2 pixCoordBase, const bool finalApply )
{
	float DenoiseBlurBeta = float(1e4f);
	if (GetOption(XE_GTAO_DENOISE_PASSES) != 0)
	{
		DenoiseBlurBeta = float(1.2f);
	}
	const float blurAmount = (finalApply)?(float(DenoiseBlurBeta)):(float(DenoiseBlurBeta)/float(5.0));
	const float diagWeight = 0.85 * 0.5;

	AOTermType aoTerm[2];   // pixel pixCoordBase and pixel pixCoordBase + int2( 1, 0 )
	float4 edgesC_LRTB[2];
	float weightTL[2];
	float weightTR[2];
	float weightBL[2];
	float weightBR[2];

	// gather edge and visibility quads, used later
	const float2 gatherCenter = float2( pixCoordBase.x, pixCoordBase.y ) * GetInvResolution();
	float4 edgesQ0        = gather_red(WORKINGEDGES_INPUT_INDEX, float3(gatherCenter, 0), int2( 0, 0 ) );
	float4 edgesQ1        = gather_red(WORKINGEDGES_INPUT_INDEX, float3(gatherCenter, 0), int2( 2, 0 ) );
	float4 edgesQ2        = gather_red(WORKINGEDGES_INPUT_INDEX, float3(gatherCenter, 0), int2( 1, 2 ) );

	AOTermType visQ0[4];    XeGTAO_DecodeGatherPartial( gather_red(PREV_PASS_OUTPUT_INPUT_INDEX, float3(gatherCenter, 0), int2( 0, 0 ) ), visQ0 );
	AOTermType visQ1[4];    XeGTAO_DecodeGatherPartial( gather_red(PREV_PASS_OUTPUT_INPUT_INDEX, float3(gatherCenter, 0), int2( 2, 0 ) ), visQ1 );
	AOTermType visQ2[4];    XeGTAO_DecodeGatherPartial( gather_red(PREV_PASS_OUTPUT_INPUT_INDEX, float3(gatherCenter, 0), int2( 0, 2 ) ), visQ2 );
	AOTermType visQ3[4];    XeGTAO_DecodeGatherPartial( gather_red(PREV_PASS_OUTPUT_INPUT_INDEX, float3(gatherCenter, 0), int2( 2, 2 ) ), visQ3 );
	
	//const int2 pixCoord = int2( pixCoordBase.x, pixCoordBase.y );
	//XeGTAO_Output( pixCoord, visQ0[0], finalApply );

	for( int side = 0; side < 2; side++ )
	{
		const int2 pixCoord = int2( pixCoordBase.x + side, pixCoordBase.y );

		//float4 edgesL_LRTB  = XeGTAO_UnpackEdges( (side==0)?(edgesQ0.x):(edgesQ0.y) );
		//float4 edgesT_LRTB  = XeGTAO_UnpackEdges( (side==0)?(edgesQ0.z):(edgesQ1.w) );
		//float4 edgesR_LRTB  = XeGTAO_UnpackEdges( (side==0)?(edgesQ1.x):(edgesQ1.y) );
		//float4 edgesB_LRTB  = XeGTAO_UnpackEdges( (side==0)?(edgesQ2.w):(edgesQ2.z) );

		//edgesC_LRTB[side]     = XeGTAO_UnpackEdges( (side==0)?(edgesQ0.y):(edgesQ1.x) );
		
		float4 edgesL_LRTB  = (side==0)?(float4(edgesQ0.x, edgesQ0.x, edgesQ0.x, edgesQ0.x)):(float4(edgesQ0.y, edgesQ0.y, edgesQ0.y, edgesQ0.y));
		float4 edgesT_LRTB  = (side==0)?(float4(edgesQ0.z, edgesQ0.z, edgesQ0.z, edgesQ0.z)):(float4(edgesQ1.w, edgesQ1.w, edgesQ1.w, edgesQ1.w));
		float4 edgesR_LRTB  = (side==0)?(float4(edgesQ1.x, edgesQ1.x, edgesQ1.x, edgesQ1.x)):(float4(edgesQ1.y, edgesQ1.y, edgesQ1.y, edgesQ1.y));
		float4 edgesB_LRTB  = (side==0)?(float4(edgesQ2.w, edgesQ2.w, edgesQ2.w, edgesQ2.w)):(float4(edgesQ2.z, edgesQ2.z, edgesQ2.z, edgesQ2.z));

		edgesC_LRTB[side]     = (side==0)?(float4(edgesQ0.y, edgesQ0.y, edgesQ0.y, edgesQ0.y)):(float4(edgesQ1.x, edgesQ1.x, edgesQ1.x, edgesQ1.x));

		// Edges aren't perfectly symmetrical: edge detection algorithm does not guarantee that a left edge on the right pixel will match the right edge on the left pixel (although
		// they will match in majority of cases). This line further enforces the symmetricity, creating a slightly sharper blur. Works real nice with TAA.
		edgesC_LRTB[side] *= float4( edgesL_LRTB.y, edgesR_LRTB.x, edgesT_LRTB.w, edgesB_LRTB.z );

#if 1   // this allows some small amount of AO leaking from neighbours if there are 3 or 4 edges; this reduces both spatial and temporal aliasing
		const float leak_threshold = 2.5; const float leak_strength = 0.5;
		float edginess = (clamp(4.0 - leak_threshold - dot( edgesC_LRTB[side], float4(1, 1, 1, 1) ), 0.0, 1.0) / (4-leak_threshold)) * leak_strength;
		edgesC_LRTB[side] = clamp( edgesC_LRTB[side] + edginess, 0.0, 1.0 );
#endif

#if XE_GTAO_SHOW_EDGES == 1
		SetAdditionalOutput(FINAL_DEBUG_IMG_OUTPUT_INDEX, 1.0 - float4( edgesC_LRTB[side].x, edgesC_LRTB[side].y * 0.5 + edgesC_LRTB[side].w * 0.5, edgesC_LRTB[side].z, 1.0 ), int3(pixCoord, 0));
		//g_outputDbgImage[pixCoord] = 1 - float4( edgesC_LRTB[side].z, edgesC_LRTB[side].w , 1, 0 );
		//g_outputDbgImage[pixCoord] = edginess.xxxx;
#endif

		// for diagonals; used by first and second pass
		weightTL[side] = diagWeight * (edgesC_LRTB[side].x * edgesL_LRTB.z + edgesC_LRTB[side].z * edgesT_LRTB.x);
		weightTR[side] = diagWeight * (edgesC_LRTB[side].z * edgesT_LRTB.y + edgesC_LRTB[side].y * edgesR_LRTB.z);
		weightBL[side] = diagWeight * (edgesC_LRTB[side].w * edgesB_LRTB.x + edgesC_LRTB[side].x * edgesL_LRTB.w);
		weightBR[side] = diagWeight * (edgesC_LRTB[side].y * edgesR_LRTB.w + edgesC_LRTB[side].w * edgesB_LRTB.y);

		// first pass
		AOTermType ssaoValue     = (side==0)?(visQ0[1]):(visQ1[0]);
		AOTermType ssaoValueL    = (side==0)?(visQ0[0]):(visQ0[1]);
		AOTermType ssaoValueT    = (side==0)?(visQ0[2]):(visQ1[3]);
		AOTermType ssaoValueR    = (side==0)?(visQ1[0]):(visQ1[1]);
		AOTermType ssaoValueB    = (side==0)?(visQ2[2]):(visQ3[3]);
		AOTermType ssaoValueTL   = (side==0)?(visQ0[3]):(visQ0[2]);
		AOTermType ssaoValueBR   = (side==0)?(visQ3[3]):(visQ3[2]);
		AOTermType ssaoValueTR   = (side==0)?(visQ1[3]):(visQ1[2]);
		AOTermType ssaoValueBL   = (side==0)?(visQ2[3]):(visQ2[2]);

		float sumWeight = blurAmount;
		AOTermType sum = ssaoValue * sumWeight;

		XeGTAO_AddSample( ssaoValueL, edgesC_LRTB[side].x, sum, sumWeight );
		XeGTAO_AddSample( ssaoValueR, edgesC_LRTB[side].y, sum, sumWeight );
		XeGTAO_AddSample( ssaoValueT, edgesC_LRTB[side].z, sum, sumWeight );
		XeGTAO_AddSample( ssaoValueB, edgesC_LRTB[side].w, sum, sumWeight );

		XeGTAO_AddSample( ssaoValueTL, weightTL[side], sum, sumWeight );
		XeGTAO_AddSample( ssaoValueTR, weightTR[side], sum, sumWeight );
		XeGTAO_AddSample( ssaoValueBL, weightBL[side], sum, sumWeight );
		XeGTAO_AddSample( ssaoValueBR, weightBR[side], sum, sumWeight );

		aoTerm[side] = sum / sumWeight;

		XeGTAO_Output( pixCoord, aoTerm[side], finalApply );

#ifdef XE_GTAO_SHOW_BENT_NORMALS
		if( finalApply )
		{
			//g_outputDbgImage[pixCoord] = float4( DisplayNormalSRGB( aoTerm[side].xyz ), 1 );
		}
#endif

	}
}

// Generic viewspace normal generate pass
float3 XeGTAO_ComputeViewspaceNormal( const uint2 pixCoord )
{
	float2 normalizedScreenPos = (pixCoord + 0.5.xx) * GetInvResolution();

	float4 valuesUL   = gather_red(DEPTH_BUFFER_INPUT_INDEX, float3( pixCoord * GetInvResolution(), 0), int2(0, 0) );
	float4 valuesBR   = gather_red(DEPTH_BUFFER_INPUT_INDEX, float3( pixCoord * GetInvResolution(), 0), int2(1, 1) );

	// viewspace Z at the center
	float viewspaceZ  = ConvertDepth(valuesUL.y);

	// viewspace Zs left top right bottom
	const float pixLZ = ConvertDepth(valuesUL.x);
	const float pixTZ = ConvertDepth(valuesUL.z);
	const float pixRZ = ConvertDepth(valuesBR.z);
	const float pixBZ = ConvertDepth(valuesBR.x);
	
	/*float viewspaceZ  = CustomToLinearDepth(valuesUL.y);
	const float pixLZ = CustomToLinearDepth(valuesUL.x);
	const float pixTZ = CustomToLinearDepth(valuesUL.z);
	const float pixRZ = CustomToLinearDepth(valuesBR.z);
	const float pixBZ = CustomToLinearDepth(valuesBR.x);*/

	/*float viewspaceZ  = DEPTH_VALUE(valuesUL.y);
	const float pixLZ = DEPTH_VALUE(valuesUL.x);
	const float pixTZ = DEPTH_VALUE(valuesUL.z);
	const float pixRZ = DEPTH_VALUE(valuesBR.z);
	const float pixBZ = DEPTH_VALUE(valuesBR.x);*/

	float4 edgesLRTB  = XeGTAO_CalculateEdges( float(viewspaceZ), float(pixLZ), float(pixRZ), float(pixTZ), float(pixBZ) );

	float3 CENTER   = XeGTAO_ComputeViewspacePosition( normalizedScreenPos, viewspaceZ );
	float3 LEFT     = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + float2(-1,  0) * GetInvResolution(), pixLZ );
	float3 RIGHT    = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + float2( 1,  0) * GetInvResolution(), pixRZ );
	float3 TOP      = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + float2( 0, -1) * GetInvResolution(), pixTZ );
	float3 BOTTOM   = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + float2( 0,  1) * GetInvResolution(), pixBZ );
	return XeGTAO_CalculateNormal( edgesLRTB, CENTER, LEFT, RIGHT, TOP, BOTTOM );
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Dolphin Passes
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Skip the depth mip sampling for now
//void FilterDepthPass(uint3 workGroupID, uint3 localInvocationID, uint3 globalInvocationID)
//{
//	XeGTAO_PrefilterDepths16x16( dispatchThreadID, groupThreadID, g_GTAOConsts, g_srcRawDepth, g_samplerPointClamp, g_outWorkingDepthMIP0, g_outWorkingDepthMIP1, g_outWorkingDepthMIP2, g_outWorkingDepthMIP3, g_outWorkingDepthMIP4 );
//}

void DepthToNormalPass(uint3 workGroupID, uint3 localInvocationID, uint3 globalInvocationID)
{
	float3 viewspaceNormal = XeGTAO_ComputeViewspaceNormal( globalInvocationID.xy );
	viewspaceNormal.y *= -1;
	viewspaceNormal = clamp( viewspaceNormal * 0.5 + 0.5, 0.0, 1.0 );
	SetOutput(float4(viewspaceNormal, 1), int3(int2(globalInvocationID.xy), 0));
}

float3 LoadNormal( float2 pixelCoords )
{
	float3 encodedNormal = SampleInputLocation(PREV_PASS_OUTPUT_INPUT_INDEX, pixelCoords).xyz;
	return normalize(encodedNormal * 2.0.xxx - 1.0.xxx);
}

// From https://www.shadertoy.com/view/3tB3z3 - except we're using R2 here
#define XE_HILBERT_LEVEL    6U
#define XE_HILBERT_WIDTH    ( (1U << XE_HILBERT_LEVEL) )
#define XE_HILBERT_AREA     ( XE_HILBERT_WIDTH * XE_HILBERT_WIDTH )
uint HilbertIndex( uint posX, uint posY )
{   
	uint index = 0U;
	for( uint curLevel = XE_HILBERT_WIDTH/2U; curLevel > 0U; curLevel /= 2U )
	{
		uint regionX = uint(( posX & curLevel ) > 0U);
		uint regionY = uint(( posY & curLevel ) > 0U);
		index += curLevel * curLevel * ( (3U * regionX) ^ regionY);
		if( regionY == 0U )
		{
			if( regionX == 1U )
			{
				posX = uint( (XE_HILBERT_WIDTH - 1U) ) - posX;
				posY = uint( (XE_HILBERT_WIDTH - 1U) ) - posY;
			}

			uint temp = posX;
			posX = posY;
			posY = temp;
		}
	}
	return index;
}

// Engine-specific screen & temporal noise loader
float2 SpatioTemporalNoise( uint2 pixCoord, uint temporalIndex )    // without TAA, temporalIndex is always 0
{
	float2 noise;
#if 1   // Hilbert curve driving R2 (see https://www.shadertoy.com/view/3tB3z3)
	#ifdef XE_GTAO_HILBERT_LUT_AVAILABLE // load from lookup texture...
		uint index = g_srcHilbertLUT.Load( uint3( pixCoord % 64, 0 ) ).x;
	#else // ...or generate in-place?
		uint index = HilbertIndex( pixCoord.x, pixCoord.y );
	#endif
	index += 288*(temporalIndex%64); // why 288? tried out a few and that's the best so far (with XE_HILBERT_LEVEL 6U) - but there's probably better :)
	// R2 sequence - see http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
	return float2( fract( 0.5 + index * float2(0.75487766624669276005, 0.5698402909980532659114) ) );
#else   // Pseudo-random (fastest but looks bad - not a good choice)
	uint baseHash = Hash32( pixCoord.x + (pixCoord.y << 15) );
	baseHash = Hash32Combine( baseHash, temporalIndex );
	return float2( Hash32ToFloat( baseHash ), Hash32ToFloat( Hash32( baseHash ) ) );
#endif
}

void MainPass(uint3 workGroupID, uint3 localInvocationID, uint3 globalInvocationID)
{
	uint noiseIndex = 0;
	if (GetOption(XE_GTAO_DENOISE_PASSES) != 0)
	{
		noiseIndex = frame_count % 64;
	}
	
	float2 pixelCoords = float2(float2(globalInvocationID.xy) * GetInvResolution().xy);

	if (GetOption(XE_GTAO_QUALITY) == 0)
	{
		XeGTAO_MainPass( globalInvocationID.xy, 1, 2, SpatioTemporalNoise(globalInvocationID.xy, noiseIndex), LoadNormal(pixelCoords) );
	}
	else if (GetOption(XE_GTAO_QUALITY) == 1)
	{
		XeGTAO_MainPass( globalInvocationID.xy, 2, 2, SpatioTemporalNoise(globalInvocationID.xy, noiseIndex), LoadNormal(pixelCoords) );
	}
	else if (GetOption(XE_GTAO_QUALITY) == 2)
	{
		XeGTAO_MainPass( globalInvocationID.xy, 3, 3, SpatioTemporalNoise(globalInvocationID.xy, noiseIndex), LoadNormal(pixelCoords) );
	}
	else if (GetOption(XE_GTAO_QUALITY) == 3)
	{
		XeGTAO_MainPass( globalInvocationID.xy, 9, 3, SpatioTemporalNoise(globalInvocationID.xy, noiseIndex), LoadNormal(pixelCoords) );
	}
}

void DenoisePass(uint3 workGroupID, uint3 localInvocationID, uint3 globalInvocationID)
{
	const uint2 pixCoordBase = globalInvocationID.xy * uint2( 2, 1 );    // we're computing 2 horizontal pixels at a time (performance optimization)
	XeGTAO_Denoise( pixCoordBase, false );
	
#if XE_GTAO_SHOW_NORMALS == 1
	float2 pixelCoords = float2(float2(globalInvocationID.xy) * GetInvResolution().xy);
	int3 output_pt = int3(int2(globalInvocationID.xy), 0);
	
	float4 debug = SampleInputLocation(DEBUG_IMG_INPUT_INDEX, pixelCoords);
	SetAdditionalOutput(FINAL_DEBUG_IMG_OUTPUT_INDEX, debug, output_pt);
#endif
}

void DenoisePassFinal(uint3 workGroupID, uint3 localInvocationID, uint3 globalInvocationID)
{
	const uint2 pixCoordBase = globalInvocationID.xy * uint2( 2, 1 );    // we're computing 2 horizontal pixels at a time (performance optimization)
	XeGTAO_Denoise( pixCoordBase, true );
	
	/*float2 pixelCoords = float2(float2(globalInvocationID.xy) * GetInvResolution().xy);
	int3 output_pt = int3(int2(globalInvocationID.xy), 0);
	
	float4 prev = SampleInputLocation(PREV_PASS_OUTPUT_INPUT_INDEX, pixelCoords);
	SetOutput(float4(prev.xyz, 1), output_pt);*/
	
#if XE_GTAO_SHOW_NORMALS == 1
	float2 pixelCoords = float2(float2(globalInvocationID.xy) * GetInvResolution().xy);
	int3 output_pt = int3(int2(globalInvocationID.xy), 0);
	
	float4 debug = SampleInputLocation(DEBUG_IMG_INPUT_INDEX, pixelCoords);
	SetAdditionalOutput(FINAL_DEBUG_IMG_OUTPUT_INDEX, debug, output_pt);
#endif
}

void CombinePass(uint3 workGroupID, uint3 localInvocationID, uint3 globalInvocationID)
{
#if XE_GTAO_SHOW_NORMALS == 1 || XE_GTAO_SHOW_EDGES == 1
	float2 pixelCoords = float2(float2(globalInvocationID.xy) * GetInvResolution().xy);
	int3 output_pt = int3(int2(globalInvocationID.xy), 0);
	
	float4 debug = SampleInputLocation(FINAL_DEBUG_IMG_INPUT_INDEX, pixelCoords);
	SetOutput(debug, output_pt);
#elif XE_GTAO_SHOW_AO_ONLY == 1
	float2 pixelCoords = float2(float2(globalInvocationID.xy) * GetInvResolution().xy);
	int3 output_pt = int3(int2(globalInvocationID.xy), 0);
	
	float4 prev = SampleInputLocation(PREV_PASS_OUTPUT_INPUT_INDEX, pixelCoords);
	SetOutput(float4(prev.xyz, 1), output_pt);
#else
	float2 pixelCoords = float2(float2(globalInvocationID.xy) * GetInvResolution().xy);
	int3 output_pt = int3(int2(globalInvocationID.xy), 0);
	
	float4 prev = SampleInputLocation(PREV_PASS_OUTPUT_INPUT_INDEX, pixelCoords);
	float4 color = SampleInputLocation(COLOR_BUFFER_INPUT_INDEX, pixelCoords);
	
	float4 final = float4(prev.rgb * color.rgb, 1);
	SetOutput(final, output_pt);
#endif
}
