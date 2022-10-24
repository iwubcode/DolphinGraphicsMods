#define BLOOM_LINEAR_COEF	((BLOOM_LINEAR_TAPS * 2) - 1)

float3 BlendScreen(float3 a, float3 b) {
	return 1 - ((1 - a) * (1 - b));
}
float3 BlendSoftLight(float3 a, float3 b) {
	return (1 - 2 * b) * pow(a, float3(2.0, 2.0, 2.0)) + 2 * b * a;
}
float3 BlendColorDodge(float3 a, float3 b) {
	return a / (1 - b);
}

void GaussBlurFirstPass()
{
	float4 ret = max(SamplePrev() - BLOOM_THRESHOLD, 0.0);
	
	//SetOutput(SamplePrev());

	float2 coords = GetCoordinates();
	for(int i=1; i < BLOOM_LINEAR_TAPS; i++)
	{
		float2 offset = float2(i * GetInvSrcResolution().x * BLOOM_RADIUS, 0.0);
		ret += max(SamplePrevLocation(coords + offset) - BLOOM_THRESHOLD, 0.0);
		ret += max(SamplePrevLocation(coords - offset) - BLOOM_THRESHOLD, 0.0);
	}
	
	SetOutput(ret / (1.0 - BLOOM_THRESHOLD) / BLOOM_LINEAR_COEF);
}

void GaussBlurH()
{
	float4 ret = SamplePrev();

	float2 coords = GetCoordinates();
	for(int i=1; i < BLOOM_LINEAR_TAPS; i++)
	{
		float2 offset = float2(i * GetInvSrcResolution().x * BLOOM_RADIUS, 0.0);
		ret += SamplePrevLocation(coords + offset);
		ret += SamplePrevLocation(coords - offset);
	}
	
	SetOutput(ret / BLOOM_LINEAR_COEF);
}

void GaussBlurV()
{
	float4 ret = SamplePrev();
	
	SetOutput(ret);
	
	float2 coords = GetCoordinates();
	for(int i=1; i < BLOOM_LINEAR_TAPS; i++)
	{
		float2 offset = float2(0.0, i * GetInvSrcResolution().y * BLOOM_RADIUS);
		ret += SamplePrevLocation(coords + offset);
		ret += SamplePrevLocation(coords - offset);
	}
	
	SetOutput(ret / BLOOM_LINEAR_COEF);
}

void Combine()
{
	float4 ret = SampleInput(PREV_SHADER_OUTPUT_INPUT_INDEX);
	float4 bloom = SamplePrev();
	float dotted = dot(bloom.rgb, float3(0.2126, 0.7152, 0.0722));
	bloom.rgb = lerp(float3(dotted, dotted, dotted), bloom.rgb, BLOOM_SATURATION) * BLOOM_STRENGTH;

	if (BLOOM_DEBUG)
	{
		SetOutput(bloom);
	}
	else
	{
		if (BLOOM_BLEND == 0) //Add
			ret.rgb += bloom.rgb;
		else if (BLOOM_BLEND == 1) //Add - No clip
			ret.rgb += bloom.rgb * saturate(1.0 - ret.rgb);
		else if (BLOOM_BLEND == 2) //Screen
			ret.rgb = BlendScreen(ret.rgb, bloom.rgb);
		else if (BLOOM_BLEND == 3) //Soft Light
			ret.rgb = BlendSoftLight(ret.rgb, bloom.rgb);
		else if (BLOOM_BLEND == 4) //Color Dodge
			ret.rgb = BlendColorDodge(ret.rgb, bloom.rgb);

		SetOutput(ret);
	}
}
