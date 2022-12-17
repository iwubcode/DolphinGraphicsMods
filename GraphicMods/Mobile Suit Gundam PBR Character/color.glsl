// TODO: Expose some variables for common shaders
#define COOK 1
#define COOK_GGX 1

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
	float invmax = 1.0 / sqrt(max(dot(T,T), dot(B,B)));
	//float invmax = inversesqrt( max( dot(T,T), dot(B,B) ) );
	return mat3( normalize(T * invmax), normalize(B * invmax), N );
}
float3 perturb_normal( float3 N, float3 P, float2 texcoord, float3 map)
{
	mat3 TBN = cotangent_frame( N, -P, texcoord);
	return normalize( TBN * map );
}

#define PI 3.1415926


// constant light position, only one light source for testing (treated as point light)
const vec4 light_pos = vec4(-2, 3, -2, 1);


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

void calculate_physical_lighting(float3 base, float metallic, float roughness, float3 N, float3 L, float3 H, float3 V, float3 light_color, float attenuation, inout float3 diffuse_color, inout float3 reflected_color)
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

	// diffuse is common for any model
	float3 diffref = (float3(1.0) - specfresnel) * phong_diffuse() * NdL;

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
#ifdef NORMAL_TEX_UNIT
		float4 map_rgb = texture(samp[NORMAL_TEX_UNIT], NORMAL_TEX_COORD);
		float4 map = map_rgb * 2.0 - 1.0;
		float3 normal = perturb_normal(normalize(data.normal), data.position, NORMAL_TEX_COORD.xy, map.xyz);
#else
		float3 normal = data.normal.xyz;
#endif

#ifdef METALLIC_TEX_UNIT
		float metallic = texture(samp[METALLIC_TEX_UNIT], METALLIC_TEX_COORD).x;
#else
		float metallic = 0;
#endif
#ifdef ROUGHNESS_TEX_UNIT
		float roughness = texture(samp[ROUGHNESS_TEX_UNIT], ROUGHNESS_TEX_COORD).x;
#else
		float roughness = 0;
#endif

		float3 eye = normalize(-data.position);
		float3 diffuse_color = float3(0, 0, 0);
		float3 reflected_color = float3(0, 0, 0);
		for (int i = 0; i < data.light_count; i++)
		{
			if (data.light[i].attenuation_type == 1)
			{
				float3 light_dir = normalize(data.light[i].position - data.position.xyz);
				float attn = (dot(normal, light_dir) >= 0.0) ? max(0.0, dot(normal, data.light[i].direction.xyz)) : 0.0;
				float3 cosAttn = data.light[i].cosatt.xyz;
				float3 distAttn = data.light[i].distatt.xyz;
				attn = max(0.0, dot(cosAttn, float3(1.0, attn, attn*attn))) / dot(distAttn, float3(1.0, attn, attn * attn));
				float3 H = normalize(light_dir + eye);
				calculate_physical_lighting(data.final_color.xyz, metallic, roughness, normal, light_dir, H, eye, data.light[i].color, attn, diffuse_color, reflected_color);
			}
			if (data.light[i].attenuation_type == 0 || data.light[i].attenuation_type == 2)
			{
				float3 light_dir = normalize(data.light[i].position - data.position.xyz);
				if (length(light_dir) == 0)
				{
					light_dir = normal;
				}
				float3 H = normalize(light_dir + eye);
				calculate_physical_lighting(data.final_color.xyz, metallic, roughness, normal, light_dir, H, eye, data.light[i].color, 1.0, diffuse_color, reflected_color);
			}
			if (data.light[i].attenuation_type == 3)
			{
				float3 light_dir = normalize(data.light[i].position - data.position.xyz);
				float distsq = dot(light_dir, light_dir);
				float dist = sqrt(distsq);
				float3 light_dir_div = light_dir / dist;
				float attn = max(0.0, dot(light_dir_div, data.light[i].direction.xyz));

				float3 cosAttn = data.light[i].cosatt.xyz;
				float3 distAttn = data.light[i].distatt.xyz;
				attn = max(0.0, cosAttn.x + cosAttn.y*attn + cosAttn.z*attn*attn) / dot( distAttn, float3(1.0, dist, distsq) );
				float3 H = normalize(light_dir + eye);
				calculate_physical_lighting(data.final_color.xyz, metallic, roughness, normal, light_dir, H, eye, data.light[i].color, attn, diffuse_color, reflected_color);
			}
		}
		diffuse_color = diffuse_color / (diffuse_color + vec3(1.0));
		diffuse_color = pow(diffuse_color, vec3(1.0/2.2)); 
		float3 result = diffuse_color * mix(data.final_color.xyz, float3(0.0), metallic) + reflected_color;
		return float4(result, 1);
	}
}
