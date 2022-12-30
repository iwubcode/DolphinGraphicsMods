// TODO: Expose some variables for common shaders
#define COOK 1
#define COOK_GGX 1

#define HEIGHT_MAP_IS_BACKWARDS 1
#define ROUGHNESS_IS_BACKWARDS 1

//#define LIGHTING_ONLY 1

#define PI 3.1415926

const float emissive_intensity = 2.5;
const float parallax_scale = 0.05;

mat3 cotangent_frame( float3 N, float3 p, float2 uv )
{
	float3 dp1 = dFdx( p );
	float3 dp2 = dFdy( p );
	float2 duv1 = dFdx( uv );
	float2 duv2 = dFdy( uv );
	float3 dp2perp = cross( dp2, N );
	float3 dp1perp = cross( N, dp1 );
	float3 T = dp2perp * duv1.x + dp1perp * duv2.x;
	float3 B = dp2perp * duv1.y + dp1perp * duv2.y;
	//float invmax = 1.0 / sqrt(max(dot(T,T), dot(B,B)));
	float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
	return mat3( normalize(T * invmax), normalize(B * invmax), N );
}
float3 perturb_normal( float3 N, float3 P, float2 texcoord, float3 map)
{
	mat3 TBN = cotangent_frame( N, -P, texcoord);
	return normalize( TBN * map );
}

vec2 ParallaxMappingTest(sampler2DArray depthMap, vec3 texCoords, vec3 viewDir, float heightScale)
{ 
	// number of depth layers
	const float minLayers = 8;
	const float maxLayers = 32;
	float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), viewDir)));
	numLayers = 32;
	// calculate the size of each layer
	float layerDepth = 1.0 / numLayers;
	// depth of current layer
	float currentLayerDepth = 0.0;
	// the amount to shift the texture coordinates per layer (from vector P)
	vec2 P = viewDir.xy * heightScale; 
	vec2 deltaTexCoords = P / numLayers;

	// get initial values
	vec2  currentTexCoords     = texCoords.xy;
#ifdef HEIGHT_MAP_IS_BACKWARDS
	float currentDepthMapValue = 1.0 - texture(depthMap, vec3(currentTexCoords, texCoords.z)).r;
#else
	float currentDepthMapValue = texture(depthMap, vec3(currentTexCoords, texCoords.z)).r;
#endif

	int loop_counter = 0;
	for( int i = 0; i < maxLayers; i++ ) {
		if ( i >= numLayers )
			break;
		if( currentLayerDepth >= currentDepthMapValue )
			break;
		// shift texture coordinates along direction of P
		currentTexCoords -= deltaTexCoords;
		// get depthmap value at current texture coordinates
#ifdef HEIGHT_MAP_IS_BACKWARDS
		currentDepthMapValue = 1.0 - texture(depthMap, vec3(currentTexCoords, texCoords.z)).r;
#else
		currentDepthMapValue = texture(depthMap, vec3(currentTexCoords, texCoords.z)).r;
#endif
		// get depth of next layer
		currentLayerDepth += layerDepth;  
		loop_counter++;
	}

	// get texture coordinates before collision (reverse operations)
	vec2 prevTexCoords = currentTexCoords + deltaTexCoords;

	// get depth after and before collision for linear interpolation
	float afterDepth  = currentDepthMapValue - currentLayerDepth;
#ifdef HEIGHT_MAP_IS_BACKWARDS
	float beforeDepth = 1.0 - texture(depthMap, vec3(prevTexCoords, texCoords.z)).r - currentLayerDepth + layerDepth;
#else
	float beforeDepth = texture(depthMap, vec3(prevTexCoords, texCoords.z)).r - currentLayerDepth + layerDepth;
#endif

	// interpolation of texture coordinates
	float weight = afterDepth / (afterDepth - beforeDepth);
	vec2 finalTexCoords = prevTexCoords * weight + currentTexCoords * (1.0 - weight);

	return finalTexCoords.xy;
}

float ShadowCalc(sampler2DArray depthMap, vec3 texCoords, vec3 lightDir, float heightScale)
{
	float minLayers = 0;
	float maxLayers = 32;
	float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), lightDir)));
	numLayers = 32;

	vec2 currentTexCoords = texCoords.xy;
#ifdef HEIGHT_MAP_IS_BACKWARDS
	float currentDepthMapValue = 1.0 - texture(depthMap, vec3(currentTexCoords, texCoords.z)).r;
#else
	float currentDepthMapValue = texture(depthMap, vec3(currentTexCoords, texCoords.z)).r;
#endif
	float currentLayerDepth = currentDepthMapValue;

	float layerDepth = 1.0 / numLayers;
	vec2 P = lightDir.xy * heightScale;
	vec2 deltaTexCoords = P / numLayers;

	for( int i = 0; i < maxLayers; i++ ) {
		if ( i >= numLayers )
			break;
		if( currentLayerDepth <= currentDepthMapValue && currentLayerDepth > 0.0 )
			break;
		// shift texture coordinates along direction of P
		currentTexCoords += deltaTexCoords;
		// get depthmap value at current texture coordinates
#ifdef HEIGHT_MAP_IS_BACKWARDS
		currentDepthMapValue = 1.0 - texture(depthMap, vec3(currentTexCoords, texCoords.z)).r;
#else
		currentDepthMapValue = texture(depthMap, vec3(currentTexCoords, texCoords.z)).r;
#endif
		// get depth of next layer
		currentLayerDepth -= layerDepth;  
	}
	
	return currentLayerDepth > currentDepthMapValue ? 0.0 : 1.0;
}

// handy value clamping to 0 - 1 range
float saturate(in float value)
{
	return clamp(value, 0.0, 1.0);
}


// phong (lambertian) diffuse term
float phong_diffuse()
{
	return (1.0 / PI);
}

// compute fresnel specular factor for given base specular and product
// product could be NdV or VdH depending on used technique
vec3 fresnel_factor(in vec3 f0, in float product)
{
	return mix(f0, vec3(1.0), pow(1.01 - product, 5.0));
}


// following functions are copies of UE4
// for computing cook-torrance specular lighting terms

float D_blinn(in float roughness, in float NdH)
{
	float m = roughness * roughness;
	float m2 = m * m;
	float n = 2.0 / m2 - 2.0;
	return (n + 2.0) / (2.0 * PI) * pow(NdH, n);
}

float D_beckmann(in float roughness, in float NdH)
{
	float m = roughness * roughness;
	float m2 = m * m;
	float NdH2 = NdH * NdH;
	return exp((NdH2 - 1.0) / (m2 * NdH2)) / (PI * m2 * NdH2 * NdH2);
}

float D_GGX(in float roughness, in float NdH)
{
	float m = roughness * roughness;
	float m2 = m * m;
	float d = (NdH * m2 - NdH) * NdH + 1.0;
	return m2 / (PI * d * d);
}

float G_schlick(in float roughness, in float NdV, in float NdL)
{
	float k = roughness * roughness * 0.5;
	float V = NdV * (1.0 - k) + k;
	float L = NdL * (1.0 - k) + k;
	return 0.25 / (V * L);
}

// simple phong specular calculation with normalization
vec3 phong_specular(in vec3 V, in vec3 L, in vec3 N, in vec3 specular, in float roughness)
{
	vec3 R = reflect(-L, N);
	float spec = max(0.0, dot(V, R));

	float k = 1.999 / (roughness * roughness);

	return min(1.0, 3.0 * 0.0398 * k) * pow(spec, min(10000.0, k)) * specular;
}

// simple blinn specular calculation with normalization
vec3 blinn_specular(in float NdH, in vec3 specular, in float roughness)
{
	float k = 1.999 / (roughness * roughness);
	
	return min(1.0, 3.0 * 0.0398 * k) * pow(NdH, min(10000.0, k)) * specular;
}

// cook-torrance specular calculation
vec3 cooktorrance_specular(in float NdL, in float NdV, in float NdH, in vec3 specular, in float roughness)
{
#ifdef COOK_BLINN
	float D = D_blinn(roughness, NdH);
#endif

#ifdef COOK_BECKMANN
	float D = D_beckmann(roughness, NdH);
#endif

#ifdef COOK_GGX
	float D = D_GGX(roughness, NdH);
#endif

	float G = G_schlick(roughness, NdV, NdL);

	float rim_intensity = 1.0;
	float rim = mix(1.0 - roughness * rim_intensity * 0.9, 1.0, NdV);

	return (1.0 / rim) * specular * G * D;
}

void calculate_physical_lighting(float3 base, float metallic, float roughness, float self_shadowing, float3 N, float3 L, float3 H, float3 V, float3 light_color, float attenuation, inout float3 diffuse_color, inout float3 reflected_color)
{
	// mix between metal and non-metal material, for non-metal
	// constant base specular factor of 0.04 grey is used
	float3 specular = mix(vec3(0.04), base, metallic);

	// diffuse IBL term
	//    I know that my IBL cubemap has diffuse pre-integrated value in 10th MIP level
	//    actually level selection should be tweakable or from separate diffuse cubemap
	//mat3x3 tnrm = transpose(normal_matrix);
	//float3 envdiff = textureCubeLod(envd, tnrm * N, 10).xyz;

	// specular IBL term
	//    11 magic number is total MIP levels in cubemap, this is simplest way for picking
	//    MIP level from roughness value (but it's not correct, however it looks fine)
	//float3 refl = tnrm * reflect(-V, N);
	//float3 envspec = textureCubeLod(
	//	envd, refl, max(roughness * 11.0, textureQueryLod(envd, refl).y)
	//).xyz;

	// compute material reflectance

	float NdL = max(0.0, dot(N, L));
	float NdV = max(0.001, dot(N, V));
	float NdH = max(0.001, dot(N, H));
	float HdV = max(0.001, dot(H, V));
	float LdV = max(0.001, dot(L, V));

	// fresnel term is common for any, except phong
	// so it will be calcuated inside ifdefs

#ifdef PHONG
	// specular reflectance with PHONG
	float3 specfresnel = fresnel_factor(specular, NdV);
	float3 specref = phong_specular(V, L, N, specfresnel, roughness);
#endif

#ifdef BLINN
	// specular reflectance with BLINN
	float3 specfresnel = fresnel_factor(specular, HdV);
	float3 specref = blinn_specular(NdH, specfresnel, roughness);
#endif

#ifdef COOK
	// specular reflectance with COOK-TORRANCE
	float3 specfresnel = fresnel_factor(specular, HdV);
	float3 specref = cooktorrance_specular(NdL, NdV, NdH, specfresnel, roughness);
#endif

	specref *= float3(NdL);
	specref = max(float3(0, 0, 0), specref);
	specref = min(float3(1, 1, 1), specref);

	// diffuse is common for any model
	float3 diffref = (float3(1.0) - specfresnel) * phong_diffuse() * NdL * (1.0 - self_shadowing);

	reflected_color += specref * light_color * attenuation;
	diffuse_color += diffref * light_color * attenuation;
}

float4 custom_main( in CustomShaderData data )
{
	if (data.texcoord_count == 0)
	{
		return data.final_color;
	}
	else
	{
		float3 eye = normalize(-data.position);

		float4 base_color = data.final_color;
		uint texmap = 0;
		if (data.tev_stage_count != 0)
		{
			uint texture_set = 0;
			for (uint i = 0; i < data.tev_stage_count; i++)
			{
				for (uint j = 0; j < 4; j++)
				{
					if (data.tev_stages[i].input_color[j].input_type == CUSTOM_SHADER_TEV_STAGE_INPUT_TYPE_TEX)
					{
						if (texture_set == 0)
						{
							base_color = float4(data.tev_stages[i].input_color[j].value, 1.0);
							texmap = data.tev_stages[i].texmap;
						}
						texture_set++;
					}
				}
			}
		}

#ifdef HEIGHT_TEX_UNIT
		mat3 TBN = cotangent_frame( data.normal, -eye, HEIGHT_TEX_COORD.xy);
		float3 eye_tangent = normalize(TBN * eye);
		float2 new_uv = ParallaxMappingTest(samp[HEIGHT_TEX_UNIT], HEIGHT_TEX_COORD, eye_tangent, parallax_scale);
		if (new_uv.x > 1.0)
			new_uv.x = 1.0;
		if (new_uv.y > 1.0)
			new_uv.y = 1.0;
		if (new_uv.x < 0.0)
			new_uv.x = 0.0;
		if (new_uv.y < 0.0)
			new_uv.y = 0.0;
		for (uint i = 0; i < 8; i++)
		{
			if (i == texmap)
			{
				base_color = texture(samp[i], float3(new_uv, 0));
			}
		}
#endif

#ifdef NORMAL_TEX_UNIT
#ifdef HEIGHT_TEX_UNIT
		float4 map_rgb = texture(samp[NORMAL_TEX_UNIT], float3(new_uv, NORMAL_TEX_COORD.z));
#else
		float4 map_rgb = texture(samp[NORMAL_TEX_UNIT], NORMAL_TEX_COORD);
#endif
		float4 map = map_rgb * 2.0 - 1.0;
#ifdef HEIGHT_TEX_UNIT
		float3 normal = perturb_normal(normalize(data.normal), eye, NORMAL_TEX_COORD.xy, map.xyz);
#else
		float3 normal = perturb_normal(normalize(data.normal), eye, NORMAL_TEX_COORD.xy, map.xyz);
#endif
#else
		float3 normal = data.normal.xyz;
#endif

#ifdef METALLIC_TEX_UNIT
#ifdef HEIGHT_TEX_UNIT
		float metallic = texture(samp[METALLIC_TEX_UNIT], float3(new_uv, METALLIC_TEX_COORD.z)).x;
#else
		float metallic = texture(samp[METALLIC_TEX_UNIT], METALLIC_TEX_COORD).x;
#endif
#else
		float metallic = 0;
#endif

#ifdef ROUGHNESS_TEX_UNIT
#ifdef HEIGHT_TEX_UNIT
		float roughness = texture(samp[ROUGHNESS_TEX_UNIT], float3(new_uv, ROUGHNESS_TEX_COORD.z)).x;
#else
		float roughness = texture(samp[ROUGHNESS_TEX_UNIT], ROUGHNESS_TEX_COORD).x;
#endif
#else
		float roughness = 0.5;
#endif

#ifdef ROUGHNESS_IS_BACKWARDS
		roughness = 1.0 - roughness;
#endif

		float3 diffuse_color = float3(0, 0, 0);
		float3 reflected_color = float3(0, 0, 0);
		for (int i = 0; i < data.light_count; i++)
		{
			if (data.light[i].attenuation_type == CUSTOM_SHADER_LIGHTING_ATTENUATION_TYPE_POINT)
			{
				float3 light_dir = normalize(data.light[i].position - data.position.xyz);
				float attn = (dot(normal, light_dir) >= 0.0) ? max(0.0, dot(normal, data.light[i].direction.xyz)) : 0.0;
				float3 cosAttn = data.light[i].cosatt.xyz;
				float3 distAttn = data.light[i].distatt.xyz;
				attn = max(0.0, dot(cosAttn, float3(1.0, attn, attn*attn))) / dot(distAttn, float3(1.0, attn, attn * attn));
				float self_shadowing = 0.0;
#ifdef HEIGHT_TEX_UNIT
				mat3 TBN = cotangent_frame( normalize(data.normal), -eye, HEIGHT_TEX_COORD.xy);
				float3 light_dir_tangent = normalize(TBN * light_dir);
				if (dot(normal, light_dir) <= 0)
				{
					self_shadowing = ShadowCalc(samp[HEIGHT_TEX_UNIT], HEIGHT_TEX_COORD, light_dir_tangent, parallax_scale);
				}
#endif
				float3 H = normalize(light_dir + eye);
				vec3 light_color = data.light[i].color;
				calculate_physical_lighting(base_color.xyz, metallic, roughness, self_shadowing, normal, light_dir, H, eye, light_color, attn, diffuse_color, reflected_color);
			}
			if (data.light[i].attenuation_type == CUSTOM_SHADER_LIGHTING_ATTENUATION_TYPE_NONE || data.light[i].attenuation_type == CUSTOM_SHADER_LIGHTING_ATTENUATION_TYPE_DIR)
			{
				float3 light_dir = normalize(data.light[i].position - data.position.xyz);
				if (length(light_dir) == 0)
				{
					light_dir = normal;
				}
				float self_shadowing = 0.0;
#ifdef HEIGHT_TEX_UNIT
				mat3 TBN = cotangent_frame( normalize(data.normal), -eye, HEIGHT_TEX_COORD.xy);
				float3 light_dir_tangent = normalize(TBN * light_dir);
				if (dot(normal, light_dir) <= 0)
				{
					self_shadowing = ShadowCalc(samp[HEIGHT_TEX_UNIT], HEIGHT_TEX_COORD, light_dir_tangent, parallax_scale);
				}
#endif
				float3 H = normalize(light_dir + eye);
				vec3 light_color = data.light[i].color;
				calculate_physical_lighting(base_color.xyz, metallic, roughness, self_shadowing, normal, light_dir, H, eye, light_color, 1.0, diffuse_color, reflected_color);
			}
			if (data.light[i].attenuation_type == CUSTOM_SHADER_LIGHTING_ATTENUATION_TYPE_SPOT)
			{
				float3 light_dir = normalize(data.light[i].position - data.position.xyz);
				float distsq = dot(light_dir, light_dir);
				float dist = sqrt(distsq);
				float3 light_dir_div = light_dir / dist;
				float attn = max(0.0, dot(light_dir_div, data.light[i].direction.xyz));

				float3 cosAttn = data.light[i].cosatt.xyz;
				float3 distAttn = data.light[i].distatt.xyz;
				attn = max(0.0, cosAttn.x + cosAttn.y*attn + cosAttn.z*attn*attn) / dot( distAttn, float3(1.0, dist, distsq) );
				float self_shadowing = 0.0;
#ifdef HEIGHT_TEX_UNIT
				mat3 TBN = cotangent_frame( normalize(data.normal), -eye, HEIGHT_TEX_COORD.xy);
				float3 light_dir_tangent = normalize(TBN * light_dir);
				if (dot(normal, light_dir) <= 0)
				{
					self_shadowing = ShadowCalc(samp[HEIGHT_TEX_UNIT], HEIGHT_TEX_COORD, light_dir_tangent, parallax_scale);
				}
#endif
				float3 H = normalize(light_dir + eye);
				vec3 light_color = data.light[i].color;
				calculate_physical_lighting(base_color.xyz, metallic, roughness, self_shadowing, normal, light_dir, H, eye, light_color, attn, diffuse_color, reflected_color);
			}
		}
		diffuse_color = diffuse_color / (diffuse_color + vec3(1.0));
		diffuse_color = pow(diffuse_color, vec3(1.0/2.2));
#ifdef LIGHTING_ONLY
		float3 result = diffuse_color;
#else
		float3 result = diffuse_color * mix(base_color.xyz, float3(0.0), metallic) + base_color.xyz * 0.5 + reflected_color;
#endif

#ifdef EMISSIVE_TEX_UNIT
#ifdef HEIGHT_TEX_UNIT
		result += texture(samp[EMISSIVE_TEX_UNIT], float3(new_uv, EMISSIVE_TEX_COORD.z)).xyz * emissive_intensity;
#else
		result += texture(samp[EMISSIVE_TEX_UNIT], EMISSIVE_TEX_COORD).xyz * emissive_intensity;
#endif
#endif

		return float4(result, 1);
	}
}
