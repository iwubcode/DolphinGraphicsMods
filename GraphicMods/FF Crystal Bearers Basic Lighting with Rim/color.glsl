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

		float3 eye = normalize(-data.position);
		float rimLightIntensity = dot(eye, normal);
		rimLightIntensity = 1.0 - rimLightIntensity;
		rimLightIntensity = max(0.0, rimLightIntensity);
		rimLightIntensity = pow(rimLightIntensity, 2);
		rimLightIntensity = smoothstep(0.3, 0.4, rimLightIntensity);
		float total_diffuse = 0;
		for (int i = 0; i < data.light_count; i++)
		{
			if (data.light[i].attenuation_type == 1)
			{
				float3 light_dir = normalize(data.light[i].position - data.position.xyz);
				float attn = (dot(normal, light_dir) >= 0.0) ? max(0.0, dot(normal, data.light[i].direction.xyz)) : 0.0;
				float3 cosAttn = data.light[i].cosatt.xyz;
				float3 distAttn = data.light[i].distatt.xyz;
				attn = max(0.0, dot(cosAttn, float3(1.0, attn, attn*attn))) / dot(distAttn, float3(1.0, attn, attn * attn));
				total_diffuse += attn * max(0.0, dot(normal, light_dir));
			}
			if (data.light[i].attenuation_type == 0 || data.light[i].attenuation_type == 2)
			{
				float3 light_dir = normalize(data.light[i].position - data.position.xyz);
				if (length(light_dir) == 0)
				{
					light_dir = normal;
				}
				total_diffuse += dot(normal, light_dir);
			}
			if (data.light[i].attenuation_type == 3)
			{
				float3 light_dir = normalize(data.light[i].position - data.position.xyz);
				float distsq = dot(light_dir, light_dir);
				float dist = sqrt(distsq);
				light_dir = light_dir / dist;
				float attn = max(0.0, dot(light_dir, data.light[i].direction.xyz));

				float3 cosAttn = data.light[i].cosatt.xyz;
				float3 distAttn = data.light[i].distatt.xyz;
				attn = max(0.0, cosAttn.x + cosAttn.y*attn + cosAttn.z*attn*attn) / dot( distAttn, float3(1.0, dist, distsq) );
				total_diffuse += attn * dot(normal, light_dir);
			}
		}
		total_diffuse = clamp(total_diffuse, 0., 1.);
		float3 rimColor = float3(1, 1, 1);
		//return data.final_color * total_diffuse + data.final_color * 0.5;
		return data.final_color * max(total_diffuse * 1.25, 1) + float4(rimLightIntensity * data.final_color.xyz, 0);
		//return float4(total_diffuse, total_diffuse, total_diffuse, 1);
	}
}
