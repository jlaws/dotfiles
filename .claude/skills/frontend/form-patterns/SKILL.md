---
name: form-patterns
description: "Use when building forms in React or Next.js. Covers React Hook Form with Zod validation, multi-step wizards, server actions, file uploads, and accessible form UX."
---

# Form Patterns

## Stack Decision

| Need | Use |
|------|-----|
| Client-side React forms | React Hook Form + Zod |
| Next.js server mutations | Server Actions + useActionState |
| Simple 1-2 field forms | Uncontrolled + native validation |
| Complex multi-step | React Hook Form + state machine |

## React Hook Form + Zod

```typescript
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'

const schema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
  role: z.enum(['admin', 'user']),
})
type FormData = z.infer<typeof schema>

function SignupForm() {
  const { register, handleSubmit, formState: { errors, isSubmitting } } = useForm<FormData>({
    resolver: zodResolver(schema),
    defaultValues: { role: 'user' },
  })

  return (
    <form onSubmit={handleSubmit(onSubmit)} noValidate>
      <div>
        <label htmlFor="email">Email</label>
        <input id="email" type="email" {...register('email')} aria-describedby="email-error" />
        {errors.email && <p id="email-error" role="alert">{errors.email.message}</p>}
      </div>
      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Submitting...' : 'Sign Up'}
      </button>
    </form>
  )
}
```

## Multi-Step Wizard

```typescript
const steps = ['profile', 'address', 'confirm'] as const
type Step = (typeof steps)[number]

function WizardForm() {
  const [step, setStep] = useState<Step>('profile')
  const [data, setData] = useState<Partial<WizardData>>({})

  function handleStepSubmit(stepData: Partial<WizardData>) {
    const merged = { ...data, ...stepData }
    setData(merged)
    const idx = steps.indexOf(step)
    if (idx < steps.length - 1) {
      setStep(steps[idx + 1])
    } else {
      submitFinal(merged as WizardData)
    }
  }

  return (
    <>
      {step === 'profile' && <ProfileStep defaults={data} onSubmit={handleStepSubmit} />}
      {step === 'address' && <AddressStep defaults={data} onSubmit={handleStepSubmit} />}
      {step === 'confirm' && <ConfirmStep data={data} onSubmit={handleStepSubmit} />}
    </>
  )
}

// Each step is its own React Hook Form instance
function ProfileStep({ defaults, onSubmit }: StepProps) {
  const { register, handleSubmit } = useForm({ defaultValues: defaults })
  return <form onSubmit={handleSubmit(onSubmit)}>...</form>
}
```

## Server Actions (Next.js)

```typescript
// action.ts
'use server'
import { z } from 'zod'

const schema = z.object({ title: z.string().min(1), content: z.string() })

export async function createPost(prevState: ActionState, formData: FormData): Promise<ActionState> {
  const parsed = schema.safeParse(Object.fromEntries(formData))
  if (!parsed.success) {
    return { errors: parsed.error.flatten().fieldErrors }
  }
  await db.posts.create({ data: parsed.data })
  revalidatePath('/posts')
  return { success: true }
}

// component.tsx
'use client'
import { useActionState } from 'react'
import { createPost } from './action'

function PostForm() {
  const [state, action, isPending] = useActionState(createPost, { errors: {} })

  return (
    <form action={action}>
      <input name="title" />
      {state.errors?.title && <p role="alert">{state.errors.title}</p>}
      <button disabled={isPending}>Create</button>
    </form>
  )
}
```

## File Upload

```typescript
function FileUpload() {
  const [preview, setPreview] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    if (file.size > 5 * 1024 * 1024) { alert('Max 5MB'); return }
    setPreview(URL.createObjectURL(file))
  }

  // Drag and drop
  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) {
      // Transfer to hidden input for form submission
      const dt = new DataTransfer()
      dt.items.add(file)
      inputRef.current!.files = dt.files
      setPreview(URL.createObjectURL(file))
    }
  }

  return (
    <div onDrop={handleDrop} onDragOver={(e) => e.preventDefault()}>
      <input ref={inputRef} type="file" accept="image/*" onChange={handleFile} />
      {preview && <img src={preview} alt="Preview" />}
    </div>
  )
}
```

## Validation UX Patterns

```typescript
// Debounced async validation (e.g., username availability)
const { register } = useForm({
  mode: 'onBlur',  // validate on blur, not every keystroke
})

register('username', {
  validate: debounce(async (value: string) => {
    const available = await checkUsername(value)
    return available || 'Username taken'
  }, 300),
})
```

**Strategy selection:**
- **Inline on blur**: best for most fields -- immediate feedback without noise
- **On submit**: better for short forms where inline feels heavy
- **Debounced async**: required for server-validated fields (uniqueness checks)

## Accessibility Checklist

- Every `<input>` has a `<label>` with matching `htmlFor`/`id`
- Error messages use `role="alert"` and link via `aria-describedby`
- Required fields use `aria-required="true"` (not just `required` attribute)
- Focus the first error field on submit failure
- Disabled submit buttons still explain why (tooltip or adjacent text)

## Gotchas

- **Controlled vs uncontrolled**: React Hook Form is uncontrolled by default; `Controller` wraps controlled components (Select, DatePicker). Don't mix `register` with `value` prop
- **FormData with checkboxes**: unchecked checkboxes are absent from FormData, not `false` -- parse with `formData.get('field') === 'on'`
- **File input reset**: setting `input.value = ''` is the only way to clear; React state won't do it
- **Server Action errors**: `useActionState` replaces `useFormState` in React 19; always return structured error objects, not thrown errors
- **Zod `.transform()`**: transforms run after validation; `z.coerce.number()` for FormData string-to-number

## Cross-References

- **frontend:react-state-management** -- managing form state alongside global state
- **frontend:nextjs-app-router-patterns** -- server actions, revalidation, and progressive enhancement
- **languages:pydantic-and-data-validation** -- server-side schema validation mirroring Zod patterns
